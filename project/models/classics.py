from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA

import json
import pandas as pd
from helpers import *


def ar_forecast(history):
    series = to_series(history)
    model = AR(series)

    model_fit = model.fit(disp=False)

    yhat = model_fit.predict(len(series), len(series)+6)
    return yhat

def arima_forecast(history):
    series = to_series(history)
    model = ARIMA(series, order=(6,0,3))

    model_fit = model.fit(disp=False)

    yhat = model_fit.predict(len(series), len(series)+6)
    return yhat

def evaluate(model_func, train, test):
    history = [x for x in train]

    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        yhat_sequence = model_func(history)
        predictions.append(yhat_sequence)
        print(test[i])
        history.append(test[i, :])

    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :], predictions)
    return score, scores

test_path = '../data/landvik/landvik_2010-2019.csv'
dataset = pd.read_csv(test_path, header=0, index_col=0, parse_dates=True, squeeze=True)

train, test = split_dataset(dataset.values)

models = dict()
models['ar'] = ar_forecast
models['arima'] = arima_forecast

days = [ ' mon ' , ' tue ' , ' wed ' , ' thr ' , ' fri ' , ' sat ', ' sun ' ]
all_scores = list()

for name, func in models.items():
    score, scores = evaluate(func, train, test)
    summarize_scores(name, score, scores)
    plt.plot(days, scores, marker= 'o' , label=name)
    print_scores(scores)
    all_scores.append(scores)

name = 'ar'
with open('../search/json/'+name+'.json', 'w') as f:
    json.dump(all_scores, f)

plt.ylabel('RMSE')
plt.legend()
plt.show()

