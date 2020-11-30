import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from math import sqrt
from sklearn.metrics import mean_squared_error
from pandas import Grouper, DataFrame, concat
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA


train_path = 'data/landvik/landvik_2010-2019.csv'
train = pd.read_csv(train_path, header=0, index_col=0, parse_dates=True, squeeze=True)

test_path = 'data/landvik/landvik_2020.csv'
test = pd.read_csv(test_path, header=0, index_col=0, parse_dates=True, squeeze=True)

train, test = train.values, test.values[0:7]


p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)

def evaluate():
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (yhat, obs))
    # evaluate forecasts
    rmse = sqrt(mean_squared_error(test, predictions))
    #print('Test RMSE: %.3f' % rmse)
    #print(p, d, q)
    return rmse, (p, d, q)

best_score = 100
best_params = None
for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                rmse, order = evaluate()
                print("rmse: %.3f" % rmse, order)
                if rmse < best_score:
                    best_score = rmse
                    best_params = order
            except:
                continue
print('Best rmse:', best_score, 'best params:', best_params)
# rmse: 2.165, order: 8,0,0 and 8,2,1
# baseline: 2.551

# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

