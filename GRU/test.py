#test script


import os
import glob
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

dataDir = 'test'

def loadData(path, scaler = 'standard'):
  csv_files = glob.glob(path+'/*.csv')
  train_x = []
  train_y = []
  l = []
  k = -1
  for file in csv_files:
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

X_test, Y_test = loadData(dataDir, scaler = 'standard')

loaded_model = load_model('model.h5')

test_loss, test_accuracy = loaded_model.evaluate(X_test, Y_test)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)