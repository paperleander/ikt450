from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA

import json
import pandas as pd
from helpers import *


# evaluate a single model
def evaluate(model_func, train, test):
    history = [x for x in train]

    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        yhat_sequence = model_func(history)
        predictions.append(yhat_sequence)
        history.append(test[i, :])

    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :], predictions)
    return score, scores

def daily_persistence(history):
    last_week = history[-1]
    yesterday = last_week[-1]
    forecast = [yesterday for _ in range(7)]
    return forecast

def weekly_persistence(history):
    last_week = history[-1]
    return last_week[:]

def week_one_year_ago_persistence(history):
    # get the data for the week one year ago
    last_week = history[-52]
    return last_week[:]

def week_average_each_year_persistence(history):
    weeks = list()
    for i in range(9):
        weeks.append(history[-52*i])
    s = 0
    for week in weeks:
        for day in week:
            s += day
    avg = s / int(len(weeks) * 7)
    return [avg for _ in range(7)]


test_path = '../data/landvik/landvik_2010-2019.csv'
dataset = pd.read_csv(test_path, header=0, index_col=0, parse_dates=True, squeeze=True)

train, test = split_dataset(dataset.values)

models = dict()
#models['daily'] = daily_persistence
#models['Last week'] = weekly_persistence
#models['Week last year'] = week_one_year_ago_persistence
models['Average Week last 9 years'] = week_average_each_year_persistence

all_scores = list()
days = [ ' mon ' , ' tue ' , ' wed ' , ' thr ' , ' fri ' , ' sat ', ' sun ' ]

for name, func in models.items():
    score, scores = evaluate(func, train, test)
    summarize_scores(name, score, scores)
    plt.plot(days, scores, marker= 'o' , label=name)
    print_scores(scores)
    all_scores.append(scores)

with open('json/simple.json', 'w') as f:
    json.dump(all_scores, f)

plt.ylim([2,6])
plt.ylabel('RMSE')
plt.legend()
plt.show()

