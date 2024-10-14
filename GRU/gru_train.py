# -*- coding: utf-8 -*-
"""prj2_train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UQDAgmKL_ZnH6dgRddA2W1isCM0Y15DZ

Mount the google drive
"""

from google.colab import drive

drive.mount('/content/drive', force_remount=True)

"""Check the contents in drive"""

dataDir = '/content/drive/MyDrive/deepLearning/training-validation/'

import os
import glob
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler, MinMaxScaler


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

def loadData(path, scaler = 'standard'):
  csv_files = glob.glob(path+'/**/*.csv')
  train_x = []
  train_y = []
  l = []
  for file in csv_files:
    # print(file)
    df = pd.read_csv(file, header=None)
    del df[df.columns[0]]
    df = df[df.iloc[:, 3]!='#']
    df = df.apply(pd.to_numeric, errors='coerce')
    for col in df.columns[0:len(df.columns)]:
      col_mode = df[col].mode()
      df[col].fillna(col_mode, inplace=True)
    l.append(df)
  c_df = pd.concat(l, ignore_index=True)
  if scaler == 'standard':
    print(f'using standardscaler for standardization')
    scaler = StandardScaler()
  else:
    print(f'using min-max-scaler for standardization')
    scaler = MinMaxScaler()
  d = c_df.iloc[:, 0:3]
  n_df = pd.DataFrame(scaler.fit_transform(d), columns=d.columns)
  #n_df = d
  res = c_df.iloc[:, 3]
  train_x = [n_df[i:i+31] for i in range(0, len(n_df), 31)]
  train_y = [res[i] for i in range(0, len(res), 31)]
  return np.array(train_x), np.array(train_y)

def splitdata(train_x, train_y):
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
    return train_x, val_x, train_y, val_y

# Build GRU model
def build_model(num_gru_layers, num_gru_units, regularization = None, kernel_initializer = 'he_normal'):
    model = Sequential()
    for _ in range(num_gru_layers-1):
        if regularization == 'l2':
            model.add(GRU(num_gru_units, return_sequences=True, dropout = 0.4, recurrent_dropout=0.4, kernel_initializer=kernel_initializer,kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        elif regularization == 'l1':
            model.add(GRU(num_gru_units, return_sequences=True, dropout = 0.4, recurrent_dropout=0.4, kernel_initializer=kernel_initializer, kernel_regularizer=tf.keras.regularizers.l1(0.01)))
        else:
            model.add(GRU(num_gru_units, return_sequences=True, dropout = 0.4 , kernel_initializer=kernel_initializer, recurrent_dropout=0.4))
    # model.add(GRU(num_gru_units,dropout = 0.6, recurrent_dropout=0.4))
    if regularization == 'l2':
        model.add(GRU(num_gru_units, dropout = 0.4, recurrent_dropout=0.4,kernel_initializer=kernel_initializer, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    elif regularization == 'l1':
        model.add(GRU(num_gru_units, dropout = 0.4, recurrent_dropout=0.4,kernel_initializer=kernel_initializer, kernel_regularizer=tf.keras.regularizers.l1(0.01)))
    else:
        model.add(GRU(num_gru_units, dropout = 0.4, kernel_initializer=kernel_initializer,recurrent_dropout=0.4))
    model.add(Dense(3, activation='softmax'))
    print(f'Model built with following parameters:-\nregularization = {regularization}\nkernel_initializer={kernel_initializer}')
    return model

# Training the model
def train_model(model, train_x, train_y, val_x, val_y, optimizer = 'adam'):
    if optimizer == 'adam':
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        op = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
        model.compile(op, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(f'Training model with optimizer = {optimizer}')
    history = model.fit(train_x, train_y, epochs=15, batch_size=32, validation_data=(val_x, val_y))
    return history

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(15)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

data = loadData(dataDir, scaler = 'standard')

train_x, val_x, train_y, val_y = splitdata(data[0], data[1])

# Build model
model = build_model(8, 32, kernel_initializer = 'glorot_uniform', regularization='l2')

# Train model
history = train_model(model, train_x, train_y, val_x, val_y)

# # Plot training history
plot_history(history)

model.save('model.h5')