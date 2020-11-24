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
from statsmodels.tsa.ar_model import AR


train_path = 'data/landvik/landvik_2010-2019.csv'
train = pd.read_csv(train_path, header=0, index_col=0, parse_dates=True, squeeze=True)

test_path = 'data/landvik/landvik_2020.csv'
test = pd.read_csv(test_path, header=0, index_col=0, parse_dates=True, squeeze=True)

test_length = 3
train, test = train.values, test.values[0:test_length+1]

model = AR(train)
fit = model.fit()
print('lag %s' % fit.k_ar)
print('CE %s' % fit.params)

predictions = fit.predict(start=len(train), end=len(train)+test_length, dynamic=False)

for i in range(len(predictions)):
    print(predictions[i], test[i])

rmse = sqrt(mean_squared_error(test, predictions))
print('rmse %.3f' % rmse)

plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

sys.exit()


