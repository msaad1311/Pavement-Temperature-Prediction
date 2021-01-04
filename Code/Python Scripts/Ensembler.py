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
import gc
import warnings
warnings.filterwarnings("ignore")

import xlwt
from xlwt import Workbook

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.decomposition import PCA


# import tensorflow as tf
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from tensorflow import keras
import tensorflow.keras.backend as K
import Models
import Train

print('Ensembler Loaded')


def calculater(paths, i, attention=False):
    appended_data = []
    errors = dict()
    for p in paths:
        flow = p.replace('\\', '/')
        df_temp = pd.read_excel(flow)
        df_temp = df_temp.loc[:, ~df_temp.columns.str.contains('^Unnamed')]
        actual = df_temp[['Actual']]

        if attention:
            df_temp = df_temp.iloc[:, 1]
        else:
            df_temp = df_temp.iloc[:, 0]

        appended_data.append(df_temp)
    df = pd.concat(appended_data, axis=1)
    df = pd.concat([df, actual], axis=1)
    df['Average'] = (df.iloc[:, i].sum(axis=1)) / len(i)
    df['Max'] = df.iloc[:, i].max(axis=1)
    df['Min'] = df.iloc[:, i].min(axis=1)
    categories = ['Average', 'Max', 'Min']
    for c in categories:
        actual = df[['Actual']]
        predict = df[[c]]
        errors[c] = metric(actual, predict)

    final = pd.DataFrame(errors).T
    final.columns = ['MAE', 'MSE', 'RMSE', 'R2', 'AGG']

    return df, final


def metric(original, predictions):
    e_mse = mse(original, predictions)
    e_rmse = sqrt(e_mse)
    e_mae = mae(original, predictions)
    e_r2 = r2(original, predictions)
    e_agg = ((e_rmse + e_mae) / 2) * (1 - e_r2)

    return (e_mae, e_mse, e_rmse, e_r2, e_agg)

def sorter(lst,n):
    return sorted([*enumerate(lst)], key=lambda x: x[1])[n][0]

def toptwo_finder(df):
    overall_mse=[]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    for m in df.columns:
        if m =='Actual':
            continue
        else:
            overall_mse.append(mse(df[['Actual']],df[[m]]))

    return [sorter(overall_mse, 0), sorter(overall_mse, 1)]

def adjuster(df,measures):
    df_temp = df.copy(deep=True)
    df_temp = np.array(df_temp)
    lst=[]
    # print(df_temp.shape)
    # print(int(np.ceil(df_temp.shape[1]/measures)))
    # print(df_temp)
    for var in range(0,df_temp.shape[1],int(np.ceil(df_temp.shape[1]/measures))+1):
        lst.append(df_temp[:,var:var+measures])
    # print(lst)
    lst_arr=np.concatenate(lst)
    lst_df = pd.DataFrame(lst_arr,columns=['MAE', 'MSE', 'RMSE', 'R2', 'AGG'])

    names = ['Average Ensemble of all with attention', 'Max Ensemble of all with attention',
             'Min Ensemble of all with attention',
             'Average Ensemble of top 2 with attention', 'Max Ensemble of top 2 with attention',
             'Min Ensemble of top 2 with attention',
             'Average Ensemble of all', 'Max Ensemble of all', 'Min Ensemble of all',
             'Average Ensemble of top 2', 'Max Ensemble of top 2', 'Min Ensemble of top 2']
    lst_temp = pd.DataFrame(names, columns=['Names'])
    lst_df = pd.concat([lst_df, lst_temp], axis=1)

    lst_df =lst_df[['Names','MAE', 'MSE', 'RMSE', 'R2', 'AGG']]

    return lst_df

def ensembler_nn(name,df,att):
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    x = df.iloc[:,0:5].values
    y = df.iloc[:,5].values

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,shuffle=False)

    y_train= y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    scaler_input = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler_input.fit_transform(x_train)
    x_test_scaled = scaler_input.transform(x_test)

    scaler_output = MinMaxScaler(feature_range=(0,1))
    y_train_scaled = scaler_output.fit_transform(y_train)
    y_test_scaled = scaler_output.transform(y_test)

    x_train_scaled = x_train_scaled.reshape(x_train_scaled.shape[0],1,x_train_scaled.shape[1])
    x_test_scaled = x_test_scaled.reshape(x_test_scaled.shape[0],1,x_test_scaled.shape[1])

    print(f'X_train shape is: {x_train_scaled.shape} and X_test shape is: {x_test_scaled.shape}')
    print(f'Y_train shape is: {y_train_scaled.shape} and Y_test shape is: {y_test_scaled.shape}')

    mlp_simple = Models.build_mlp(x_train_scaled.shape)

    Train.model_fit(f'{name}', df, 'Ensemble_new', mlp_simple, x_train_scaled, x_test_scaled,
                    y_train_scaled, y_test_scaled, 512, 500, 'Actual', scaler_output,att)
    del(mlp_simple)

    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()


def ensembler(models):
    paths=[]
    results =[]
    for i in range(len(models)):
        paths.append(os.getcwd()+ '\\'+models[i]+'\\Forecasting Values.xls')

    i=0
    names =['Attention Ensemble','Simple Ensemble']
    ## Calculating the ensemble for all the models
    for att in [True,False]:
        print(f'Working on the case when attention is {att}')
        dataframe,result=calculater(paths,[i for i in range(len(models))],att)
        # return dataframe
        ensembler_nn(names[i],dataframe,att)
        results.append(result)
        # print(dataframe.columns)
        idx = len(models)+1
        top2 = toptwo_finder(dataframe.iloc[:,:idx])

        _,result_top2 = calculater(paths,top2,att)
        results.append(result_top2)
        i=i+1

    # Train.cleaner('Ensemble', names)

    # final_results = pd.concat(results, axis=1)
    final_results = adjuster(pd.concat(results,axis=1),5)
    final_results.to_excel('Ensemble Results_new.xls')
    display(final_results)




