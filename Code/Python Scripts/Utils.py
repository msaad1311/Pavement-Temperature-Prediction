# region libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import random
from math import sqrt
import time
from IPython.display import display, HTML
from prettytable import PrettyTable
import warnings

warnings.filterwarnings("ignore")

import xlwt #pylint: disable=E401
from xlwt import Workbook #pylint: disable=E401

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

from tensorflow import keras
import tensorflow.keras.backend as K

# endregion

# region utils

def read_data(source,filename):
    df_temp = pd.read_excel(os.path.join(source,filename))
    df_temp = df_temp.loc[:, ~df_temp.columns.str.contains('^Unnamed')]
    print(f'The shape of the dataframe is {df_temp.shape}')
    print(df_temp.head())
    return df_temp


def create_dataset(X, y, time_steps, ts_range):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - ts_range):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.values[(i + time_steps):(i + time_steps + ts_range),0])
        # ys.append(y.values[(i + time_steps + ts_range-1), 0])
    return np.array(Xs), np.array(ys)

def splitter(df,output,lag,duration,ts,scale=True):
    assert (0. <= ts <= 1.)
    train_size = int(len(df) * ts)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df[train_size:]
    print(train.shape, test.shape)
    if scale:
        print('output',output)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler_single = MinMaxScaler(feature_range=(0, 1))

        scaler.fit(train)
        scaler_single.fit(train[output])

        train_scaled = pd.DataFrame(scaler.transform(train), columns=[df.columns])

        test_scaled = pd.DataFrame(scaler.transform(test), columns=[df.columns])

        df_train = train_scaled.copy(deep=True)
        df_test = test_scaled.copy(deep=True)

        # display(df_train.describe())
        # display(df_test.describe())

        x_train,y_train = create_dataset(df_train,df_train[[output]],lag,duration)
        x_test, y_test = create_dataset(df_test, df_test[[output]], lag, duration)

        # y_train = y_train. pe(-1,1)

        return x_train,x_test,y_train,y_test,scaler_single
    else:
        x_train, y_train = create_dataset(train, train[output], lag, duration)
        x_test, y_test = create_dataset(test, test[output], lag, duration)

        return x_train,x_test,y_train,y_test,None


print('Utils Loaded')

# endregion
