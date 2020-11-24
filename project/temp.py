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
series = pd.read_csv(train_path, header=0, index_col=0, parse_dates=True, squeeze=True)

X = series.values
train_size = int(len(X) * 0.9)
train, test = X[0:train_size], X[train_size:]

p, d, q = 8, 2, 1

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
        print('predicted=%f, expected=%f' % (yhat, obs))
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse, predictions


rmse, predictions = evaluate()
print("rmse: %.3f" % rmse)
# rmse: 2.165, order: 8,0,0 and 8,2,1
# rmse baseline:                2.551
# rmse (8,0,0) (2020, 1 week):  2.165
# rmse (8,2,1) (2019):          2.005

# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

