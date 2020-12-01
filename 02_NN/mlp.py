import warnings
warnings.filterwarnings("ignore")

import os
import sys
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import tensorflow as tf
from sklearn.metrics import confusion_matrix

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

df = pd.read_table("data/ecoli.data",
                   header=None,
                   delim_whitespace=True,
                   names=["seq", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "label"])

# df.hist(figsize=(15, 12))
# plt.show()

df_clean = df[(df.label == "im") | (df.label == "cp")]
print(df_clean)

X = df_clean.drop(['label', 'seq'], axis=1).to_numpy()
y = df_clean.label.replace({'cp': 0, 'im': 1}).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
        shuffle=True, random_state=42)

epochs = 250

def build_model():
    model = Sequential()
    model.add(Dense(7, input_dim=len(X[0])))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam', metrics=['accuracy'])
    #model.compile(loss= 'mse' , optimizer= 'adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0)
    return model, history

model, history = build_model()

results = model.evaluate(X_test, y_test, verbose=1)
print(results)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

sys.exit()

