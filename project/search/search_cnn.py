# naive forecast strategies for the power usage dataset
import warnings
warnings.filterwarnings("ignore")

from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

import tensorflow as tf
import pandas as pd
import os

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def split_dataset(data):
    train, test = data[5:-364], data[-364:]
    train = array(split(train, len(train)/7))
    test = array(split(test, len(test)/7))
    return train, test

def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        rmse = sqrt(mse)
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores

def summarize_scores(name, score, scores):
    s_scores = ' - ' .join(['%.1f' % s for s in scores])
    print(' %s: [%.3f] %s ' % (name, score, s_scores))

def to_series(data):
    series = [week[:] for week in data]
    series = array(series).flatten()
    return series

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
    data = train.reshape((train.shape[0]*train.shape[1], 1))
    X, y = list(), list()
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        in_start += 1
    return array(X), array(y)

# build and train model
def build_model(train, n_input, params):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 0, 20, 4
    n_timesteps = train_x.shape[1]
    n_features = train_x.shape[2]
    n_outputs = train_y.shape[1]
    filters, units = params[0], params[1]

    # define model
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=3,
        activation= 'relu' ,input_shape=(n_timesteps,n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units, activation= 'relu' ))
    model.add(Dense(n_outputs))
    model.compile(loss= 'mse' , optimizer= 'adam' )
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

def forecast(model, history, n_input):
    data = array(history)
    data = data.reshape((data.shape[0]*data.shape[1], 1))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat

def evaluate_cnn(train, test, n_input, params):
    # build and fit model
    model = build_model(train, n_input, params)
    history = [x for x in train]

    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_input)
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :], predictions)
    return score, scores


# load data
test_path = '../data/landvik/landvik_2010-2019.csv'
dataset = pd.read_csv(test_path, header=0, index_col=0, parse_dates=True, squeeze=True)

# split into train and test
train, test = split_dataset(dataset.values)

#all_inputs = [2912]
n_input = 112
averages = list()
filters = [8, 16, 32, 64, 128, 256]
units = [7, 14, 21, 28, 56, 112]

for filt in filters:
    for unit in units:
        all_scores = list()

        for i in range(5):
            score, scores = evaluate_cnn(train, test, n_input, (filt, unit))
            all_scores.append(score)
            name = 'cnn'
            summarize_scores(name, score, scores)

        avg = sum(all_scores)/len(all_scores)
        averages.append(avg)
        minimum = min(all_scores)
        maximum = max(all_scores)
        print("Filt:", filt)
        print("Unit:", unit)
        print("avg: %.3f" % avg)
        print("min: %.3f" % minimum)
        print("max: %.3f" % maximum)
        print("-"*20)

print("all_inputs:", all_inputs)
print("averages:", averages)
days = [ ' mon ' , ' tue ' , ' wed ' , ' thr ' , ' fri ' , ' sat ', ' sun ' ]
#pyplot.plot(days, scores, marker= 'o' , label=name)
pyplot.plot(all_inputs, averages)
pyplot.legend()
pyplot.show()

