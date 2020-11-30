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

def build_model(train, n_input, params, bs):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1]))
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1]))

    # define parameters
    units_1, units_2 = params[0], params[1]
    verbose, epochs, batch_size = 0, 21, 7
    n_timesteps = train_x.shape[1]
    n_outputs = train_y.shape[1]

    # define model
    model = Sequential()
    model.add(Dense(units_1, activation='relu', input_dim=n_input))
    model.add(Dense(n_outputs))
    model.compile(loss= 'mse' , optimizer= 'adam' )

    # fit network
    model.fit(train_x, train_y, epochs=bs, batch_size=batch_size, verbose=verbose)
    return model

def forecast(model, history, n_input):
    data = array(history)
    data = data.reshape((data.shape[0]*data.shape[1], 1))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    input_x = input_x.reshape((1, len(input_x)))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat

def evaluate(train, test, n_input, params, bs):
    # build and fit model
    model = build_model(train, n_input, params, bs)
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


test_path = '../data/landvik/landvik_2010-2019.csv'
dataset = pd.read_csv(test_path, header=0, index_col=0, parse_dates=True, squeeze=True)

train, test = split_dataset(dataset.values)

name = 'mlp'
n_input = 112
averages = list()
dense_1 = [300]
dense_2 = [0]
batch_size = [i*7 for i in range(1, 16)]

for d_1 in dense_1:
    for d_2 in dense_2:
        for bs in batch_size:
            all_scores = list()

            for i in range(5):
                score, scores = evaluate(train, test, n_input, (d_1, d_2), bs)
                all_scores.append(score)
                summarize_scores(name, score, scores)

            avg = sum(all_scores)/len(all_scores)
            averages.append(avg)
            minimum = min(all_scores)
            maximum = max(all_scores)
            print("dense_1:", dense_1)
            print("dense_2:", dense_2)
            #print("batch_size:", bs)
            print("epoch:", bs)
            print("avg: %.3f" % avg)
            print("min: %.3f" % minimum)
            print("max: %.3f" % maximum)
            print("-"*20)

print("averages:", averages)
days = [ ' mon ' , ' tue ' , ' wed ' , ' thr ' , ' fri ' , ' sat ', ' sun ' ]

pyplot.plot(days, averages)
pyplot.legend()
pyplot.show()

