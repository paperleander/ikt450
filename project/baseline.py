# PLOTS
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
# https://seklima.met.no/observations/

path = 'data/landvik/landvik_2005_2020.csv'
df = pd.read_csv(path, sep=";", decimal=",", encoding="UTF-8",na_values='-')

# Create another column with correct Datetime
df["Date"] = pd.to_datetime(df['Time'], format='%d.%m.%Y')

# Filter dates between 01.01.2010 to 01.01.2019 (10 years)
start_date = pd.to_datetime('01-01-2010')
end_date = pd.to_datetime('31-12-2019')
df = df[df.Date.between(start_date, end_date)]

# Set the new Datetime as the index
df.set_index('Date', inplace=True)

# Only work with Temperatur, drop other columns
df.drop(columns=["Time", "Temp(Max)", "Temp(Min)", "Wind(Max)", "Wind(Mid)"], inplace=True)

print(df.head())

# Fill in missing values
#df.interpolate() # Fill missing values
#df.reset_index()

# Get date for saving files
now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d_%H:%M:%S")


# create lagged dataset
values = DataFrame(df.values)
dataframe = concat([values.shift(1), values], axis=1)
print(dataframe.head(5))
dataframe.columns = [ ' t ' , ' t+1 ' ]
print(dataframe.head(5))

# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.90)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# persistence model
def model_persistence(x):
    return x

# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)

rmse = sqrt(mean_squared_error(test_y, predictions))
print( ' Test RMSE: %.3f ' % rmse)

# plot predictions and expected results
plt.plot(train_y)
plt.plot([None for i in train_y] + [x for x in test_y])
plt.plot([None for i in train_y] + [x for x in predictions])
plt.show()


