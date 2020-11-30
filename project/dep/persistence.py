import sys
from math import sqrt
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
from pandas import Grouper, DataFrame, concat
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot


train_path = 'data/landvik/landvik_2010-2019.csv'
train = pd.read_csv(train_path, header=0, index_col=0, parse_dates=True, squeeze=True)

test_path = 'data/landvik/landvik_2020.csv'
test = pd.read_csv(test_path, header=0, index_col=0, parse_dates=True, squeeze=True)

#print(test.head())
#sys.exit()

values = DataFrame(train.values)
train_lag = concat([values.shift(1), values], axis=1)
train_lag.columns = ['t', 't+1']
X = train_lag.values

values = DataFrame(test.values)
test_lag = concat([values.shift(1), values], axis=1)
test_lag.columns = ['t', 't+1']
Y = test_lag.values

test_size = 7

train_X, train_y = X[1:,0], X[1:,1]
test_X, test_y= Y[1:test_size+1,0], Y[1:test_size+1,1]

print(train_X, train_y)
print(test_X, test_y)
print(len(test_X), len(test_y))
#sys.exit()

# persistence model
def get_baseline(x):
    return x

# walk-forward validation
baseline= list()
for x in test_X:
    yhat = get_baseline(x)
    baseline.append(yhat)
rmse = sqrt(mean_squared_error(test_y, baseline))
print('Test RMSE: %.3f' % rmse)

# plot predictions vs expected
plt.plot(test_y)
plt.plot(baseline, color= 'red' )
plt.show()

