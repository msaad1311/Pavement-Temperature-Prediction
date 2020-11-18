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

def read_data(filepath, heading_num, row_num, rev_col, need_col, fill_miss=True):
    need_col =['Date']+need_col
    file = pd.read_excel(filepath)
    cols = file.iloc[heading_num]
    file = file.iloc[row_num:, :].reset_index()
    file.drop(columns=['index'], inplace=True)
    file.columns = cols
    file_temp = file[rev_col]
    before_fill = file_temp.isnull().sum().sum()
    if fill_miss:
        print('Filling up missing values')
        for col in file_temp.columns:
            for index, row in file_temp[[col]].iterrows():
                if pd.isnull(file_temp[col][index]):
                    file_temp[col][index] = file_temp[col][index - 365]
            print(f'Completed the {col}')
        after_fill = file_temp.isnull().sum().sum()
        print('Missing values filled')
        print(f'Missing values before: {before_fill} & Missing values after: {after_fill}')
    file_temp.columns = need_col
    file_temp[[need_col[1]]] = file_temp[[need_col[1]]].astype('float')

    file_temp.drop(file_temp.columns[0], axis=1,inplace=True)

    return file_temp


def create_dataset(X, y, time_steps, ts_range):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - ts_range):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        # ys.append(y.values[(i + time_steps):(i + time_steps + ts_range), 0])
        ys.append(y.values[(i + time_steps + ts_range-1), 0])
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

        y_train = y_train.reshape(-1,1)
        y_test = y_test.reshape(-1,1)

        return x_train,x_test,y_train,y_test,scaler_single
    else:
        x_train, y_train = create_dataset(train, train[output], lag, duration)
        x_test, y_test = create_dataset(test, test[output], lag, duration)

        return x_train,x_test,y_train,y_test,None


print('Utils Loaded')

# endregion
