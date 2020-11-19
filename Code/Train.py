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
import gc

warnings.filterwarnings("ignore")

import xlwt #pylint: disable=E401
from xlwt import Workbook #pylint: disable=E401

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

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
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from tensorflow import keras
import tensorflow.keras.backend as K
from Models import *

print('Train Loaded')


# endregion

def preliminary(name):
    filepath_simple = '{}.hdf5'.format(name)
    filepath_attention = f"{name}.hdf5"

    checkpoint_simple = keras.callbacks.ModelCheckpoint(filepath_simple,monitor='val_loss',verbose=0,save_best_only=True,mode=min)
    checkpoint_attention = keras.callbacks.ModelCheckpoint(filepath_attention, monitor='val_loss', verbose=0,save_best_only=True, mode=min)

    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0004, patience=3)

    return checkpoint_simple,checkpoint_attention,filepath_simple,filepath_attention

def plots(name,history,y_pred,y_test):
    plt.figure(figsize=(14, 9))
    plt.plot(history.history['loss'], color="red", label='Training Loss')
    plt.plot(history.history['val_loss'], color='blue', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{name} Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{name} Training and Validation Loss.png')
    plt.show()

    plt.figure(figsize=(14, 9))
    plt.plot(y_test.reshape(-1, 1), color="red", label='Actual')
    plt.plot(y_pred.reshape(-1, 1), color='blue', label='Predicted')
    plt.ylabel('Temperature')
    plt.xlabel('Time')
    plt.title(f'{name} Actual vs Predicted')
    plt.legend()
    plt.savefig(f'{name} Actual vs Predicted.png')
    plt.show()

    return

def metric(y_test,y_pred):
    i=0
    e_mae = mae(y_test,y_pred)

    e_mse = mse(y_test,y_pred)

    e_r2 = r2_score(y_test,y_pred)

    e_agg = ((np.sqrt(e_mse) + e_mae) / 2) * (1 - e_r2)

    return e_mae,e_mse,e_r2,e_agg

def cleaner(direct,name):
    path_old = os.getcwd()
    path_new = os.getcwd() + '\\{0}'.format(direct)

    os.chdir(path_new)

    df_simple = pd.read_excel(name[0]+' Results.xls',sheet_name='Simple')
    df_att = pd.read_excel(name[1]+' Results.xls',sheet_name='Attention')

    df_all = pd.concat([df_simple,df_att],axis=0)
    display(df_all)
    df_all.columns = ['Attention','MAE','MSE','RMSE','R2','AGG','Time Taken','Hours Ahead']

    df_all.to_excel('Forecasting Results.xls')

    # pred_simple = pd.read_excel(name[0]+' Results.xls',sheet_name='Predictions')
    # pred_att = pd.read_excel(name[1]+' Results.xls',sheet_name='Predictions')


    # pred_all = pd.concat([pred_simple,pred_att],axis=1)
    # pred_all.drop(columns=['Actual'],inplace=True)
    # pred_all.drop(pred_all.columns[pred_all.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    # pred_all=pd.concat([pred_all,pred_simple[['Actual']]],axis=1)

    # pred_all.to_excel('Forecasting Values.xls')

    os.remove(name[0]+' Results.xls')
    os.remove(name[1]+' Results.xls')

    print('Files deleted')

    os.chdir(path_old)

    return

def model_fit(name, direct, model, x_train, x_test, y_train, y_test, bsize, epoch,scale=None,atten=False):
    os.environ['PYTHONHASHSEED'] = '42'
    np.random.seed(42)
    # tf.random.set_seed(seed=42)
    np.random.RandomState(42)
    random.seed(42)
    path_old = os.getcwd()
    path_new = os.getcwd() + '\\{0}'.format(direct)

    wk=Workbook()
    sheet1 = wk.add_sheet('Simple', cell_overwrite_ok=True)
    sheet2 = wk.add_sheet('Attention', cell_overwrite_ok=True)
    sheet3 = wk.add_sheet('Predictions', cell_overwrite_ok=True)
    try:
        os.mkdir(path_new)
    except OSError:
        print("The directory is already created")
    else:
        print('Directory Created!!!')

    os.chdir(path_new)

    if atten:
        x = sheet2
        col = 1
        _,checkpoint,_,filepath=preliminary(name)
    else:
        x = sheet1
        col = 0
        checkpoint,_,filepath,_ = preliminary(name)

    print('X',x)

    start_time = time.time()
    history = model.fit(x_train, y_train, batch_size=bsize, epochs=epoch, validation_split=0.1, callbacks=[checkpoint])
    end_time = time.time()

    duration = round((end_time - start_time) / 60, 2)

    model.load_weights(filepath)

    y_pred = model.predict(x_test)

    if scale==None:
        y_pred_unscaled = y_pred
        y_test_unscaled = y_test
    else:
        y_pred_unscaled = scale.inverse_transform(y_pred)
        y_test_unscaled = scale.inverse_transform(y_test)
        
    print(f'The shape of y_pred is {y_pred.shape} and the shape of y_test is {y_test.shape}')

    plots(name,history,y_pred_unscaled[:,5],y_test_unscaled[:,5])


    pt = PrettyTable()

    pt.field_names = ['Hours Ahead', 'R2 Score', 'MAE', 'MSE', 'RMSE', 'AGM']

    for i in range(6):
        e_mae,e_mse,e_r2,e_agg = metric(y_test_unscaled[:,i],y_pred_unscaled[:,i])
        # i = 0
        x.write(0, 0, 'Attention')
        x.write(0, 1, 'MAE')
        x.write(0, 2, 'MSE')
        x.write(0, 3, 'RMSE')
        x.write(0, 4, 'R2')
        x.write(0, 5, 'Agg')
        x.write(0, 6, 'Time Taken')
        x.write(0,7,'Hours Ahead')
        x.write(i + 1, 0, atten)
        x.write(i + 1, 1, e_mae)
        x.write(i + 1, 2, e_mse)
        x.write(i + 1, 3, sqrt(e_mse))
        x.write(i + 1, 4, e_r2)
        x.write(i + 1, 5, e_agg)
        x.write(i + 1, 6, duration)
        x.write(i+1,7,i+1)

        pt.add_row([i + 1, round(e_r2, 2), round(e_mae, 2), round((e_mse) , 2),round((sqrt(e_mse)), 2), round(e_agg, 2)])

    # sheet3.write(0, col, name)
    # sheet3.write(0, 2, 'Actual')
    # for row, pred in enumerate(y_pred_unscaled):
    #     sheet3.write(row + 1, col, pred.item())
    #     sheet3.write(row + 1, 2, y_test_unscaled[row].item())

    

    print(pt)

    wk.save(f'{name} Results.xls')

    os.chdir(path_old)

    return y_pred_unscaled


def model_save(direct, models, names):
    path_old = os.getcwd()
    path_new = os.getcwd() + '\\{0}'.format(direct)

    os.chdir(path_new)

    for index, model in enumerate(models):
        json_model = model.to_json()

        with open(f'{names[index]}.json', 'w') as json_file:
            json_file.write(json_model)

    os.chdir(path_old)

    return


def build_model(name,scaler,x_train,x_test,y_train,y_test,batch_size,epochs):
    if name=='LSTM':
        print('You Selected LSTM')

        names = ['Simple_LSTM', 'Attention_LSTM']

        print(f'X_train shape is: {x_train.shape} and X_test shape is: {x_test.shape}')
        print(f'Y_train shape is: {y_train.shape} and Y_test shape is: {y_test.shape}')

        lstm_simple=build_lstm(x_train.shape,False)

        model_fit(names[0],name,lstm_simple,x_train,x_test,y_train,y_test,batch_size,epochs,scaler,False)

        # K.clear_session()

        print('-'*80)

        print('Moving onto Attention model')

        lstm_att = build_lstm(x_train.shape,True)

        model_fit(names[1], name, lstm_att, x_train, x_test, y_train, y_test, batch_size, epochs, scaler,True)

        lstm_models = [lstm_simple, lstm_att]
        cleaner(name,names)
        model_save(name, lstm_models, names)

        del(lstm_simple)
        del(lstm_att)

        gc.collect()
        K.clear_session()

    if name=='ConvLSTM':
        print('You selected ConvLSTM')
        names = ['Simple_ConvLSTM', 'Attention_ConvLSTM']

        x_train_convlstm = x_train.reshape(x_train.shape[0], 1, 1, x_train.shape[1], x_train.shape[2])
        x_test_convlstm = x_test.reshape(x_test.shape[0], 1, 1, x_test.shape[1], x_test.shape[2])

        print(f'X_train shape is: {x_train_convlstm.shape} and X_test shape is: {x_test_convlstm.shape}')
        print(f'Y_train shape is: {y_train.shape} and Y_test shape is: {y_test.shape}')

        convlstm_simple = build_convlstm(x_train_convlstm.shape,False)

        model_fit(names[0], name, convlstm_simple, x_train_convlstm, x_test_convlstm, y_train, y_test, batch_size, epochs,scaler,False)

        # K.clear_session()

        print('-' * 30)

        print('Moving onto Attention model')

        convlstm_att = build_convlstm(x_train_convlstm.shape, True)

        model_fit(names[1], name, convlstm_att, x_train_convlstm, x_test_convlstm, y_train, y_test, batch_size, epochs,scaler, True)

        convlstm_models = [convlstm_simple, convlstm_att]
        cleaner(name, names)
        model_save(name, convlstm_models, names)

        del(convlstm_simple)
        del(convlstm_att)

        gc.collect()
        K.clear_session()

    if name == 'CNN-LSTM':
        print('You selected CNN-LSTM')
        names = ['Simple_CNNLSTM', 'Attention_CNNLSTM']

        print(f'X_train shape is: {x_train.shape} and X_test shape is: {x_test.shape}')
        print(f'Y_train shape is: {y_train.shape} and Y_test shape is: {y_test.shape}')

        cnnlstm_simple = build_cnnlstm(x_train.shape, False)

        model_fit(names[0], name, cnnlstm_simple, x_train, x_test, y_train, y_test, batch_size, epochs,scaler,False)
        
        # K.clear_session()
        
        print('-' * 30)

        print('Moving onto Attention model')

        cnnlstm_att = build_cnnlstm(x_train.shape, True)

        model_fit(names[1], name, cnnlstm_att, x_train, x_test, y_train, y_test, batch_size, epochs,scaler, True)

        cnnlstm_models = [cnnlstm_simple, cnnlstm_att]
        cleaner(name, names)
        model_save(name, cnnlstm_models, names)

        del(cnnlstm_simple)
        del(cnnlstm_att)

        gc.collect()
        K.clear_session()

    if name =='Seq2Seq':
        print('You selected Encoder-Decorder Network')
        names = ['Simple_Seq2Seq', 'Attention_Seq2Seq']

        y_train_seq = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
        y_test_seq = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

        print(f'X_train shape is: {x_train.shape} and X_test shape is: {x_test.shape}')
        print(f'Y_train shape is: {y_train_seq.shape} and Y_test shape is: {y_test_seq.shape}')

        # print(x_train.shape[0])
        # print(x_train.shape[1])
        # print(x_train.shape[2])
        # print()
        # print(y_train_seq.shape[0])
        # print(y_train_seq.shape[1])
        # print(y_train_seq.shape[2])

        overall_shape = [x_train.shape[0],x_train.shape[1],x_train.shape[2],y_train_seq.shape[0],y_train_seq.shape[1],y_train_seq.shape[2]]

        encdec_simple = build_seq2seq(overall_shape,False)

        model_fit(names[0], name, encdec_simple, x_train, x_test, y_train_seq, y_test_seq, batch_size, epochs,scaler, False)
        
        # K.clear_session()
        
        print('-' * 30)

        print('Moving onto Attention model')

        encdec_att = build_seq2seq(overall_shape, True)

        model_fit(names[1], name, encdec_att, x_train, x_test, y_train_seq, y_test_seq, batch_size, epochs,scaler,True)

        seq2seq_models = [encdec_simple, encdec_att]

        cleaner(name, names)
        model_save(name, seq2seq_models, names)

        del(encdec_simple)
        del(encdec_att)

        gc.collect()
        K.clear_session()
        
    if name =='Wavenet':
        print('You selected WaveNet')
        names = ['Simple_Wavenet','Attention_Wavenet']

        print(f'X_train shape is: {x_train.shape} and X_test shape is: {x_test.shape}')
        print(f'Y_train shape is: {y_train.shape} and Y_test shape is: {y_test.shape}')

        wavenet_simple = build_wavenet(x_train.shape, False)

        model_fit(names[0], name, wavenet_simple, x_train, x_test, y_train, y_test, batch_size, epochs,scaler,False)
        
        # K.clear_session()
        
        print('-' * 30)

        print('Moving onto Attention model')

        wavenet_att = build_wavenet(x_train.shape, True)

        model_fit(names[1], name, wavenet_att, x_train, x_test, y_train, y_test, batch_size, epochs,scaler, True)

        wavenet_models = [wavenet_simple, wavenet_att]
        cleaner(name, names)
        model_save(name, wavenet_models, names)

        del(wavenet_simple)
        del(wavenet_att)

        gc.collect()
        K.clear_session()


    return





