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

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

print('Models Loaded')

class attention(keras.layers.Layer):
    '''
    if return_sequences=True, it will give 3D vector and if false it will give 2D vector. It is same as LSTMs.

    https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137
    the  following code is being copied from the above link.
    '''

    def __init__(self, return_sequences=True, **kwargs):
        self.return_sequences = return_sequences
        super(attention, self).__init__()

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)


def build_lstm(shape,attent):
    K.clear_session()
    model = keras.Sequential()
    model.add(keras.layers.LSTM(32, return_sequences=True, input_shape=(shape[1], shape[2])))
    model.add(keras.layers.LSTM(32, return_sequences=True))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.LSTM(32, return_sequences=True))
    model.add(keras.layers.LSTM(32, return_sequences=True))
    if attent:
        model.add(attention(return_sequences=True))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dense(6))

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])

    model.summary()

    return model

def build_convlstm(shape,attent):
    K.clear_session()
    model = keras.Sequential()
    model.add(keras.layers.ConvLSTM2D(64, kernel_size=(1, 3), return_sequences=True,
                                      input_shape=(shape[1], shape[2], shape[3], shape[4])))

    model.add(keras.layers.ConvLSTM2D(64, kernel_size=(1, 3), return_sequences=True))
    model.add(keras.layers.ConvLSTM2D(64, kernel_size=(1, 3), return_sequences=True))
    model.add(keras.layers.ConvLSTM2D(64, kernel_size=(1, 3), return_sequences=True))
    if attent:
        model.add(attention(return_sequences=True))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dense(6))

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['mae'])
    model.summary()

    return model

def build_cnnlstm(shape,attent):
    K.clear_session()
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(64, kernel_size=3, input_shape=(shape[1],shape[2])))
    model.add(keras.layers.Conv1D(64, kernel_size=3))
    model.add(keras.layers.LSTM(64, return_sequences=True))
    model.add(keras.layers.LSTM(64, return_sequences=True))
    if attent:
        model.add(attention(return_sequences=True))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dense(6))

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['mae'])

    model.summary()

    return model

def build_seq2seq(shape,attent):
    if attent:
        K.clear_session()

        input_train = keras.layers.Input(shape=(shape[1], shape[2]))
        output_train = keras.layers.Input(shape=(shape[4], shape[5]))

        ## Encoder Section##
        encoder_first = keras.layers.LSTM(64, return_sequences=True, return_state=False)(input_train)
        encoder_second = keras.layers.LSTM(64, return_sequences=True)(encoder_first)
        encoder_third = keras.layers.LSTM(64, return_sequences=True)(encoder_second)
        encoder_fourth, encoder_fourth_s1, encoder_fourth_s2 = keras.layers.LSTM(64,return_sequences=True,return_state=True)(encoder_third)

        ##Decoder Section##
        decoder_first = keras.layers.RepeatVector(output_train.shape[1])(encoder_fourth_s1)
        decoder_second = keras.layers.LSTM(64, return_state=False, return_sequences=True)(decoder_first, initial_state=[encoder_fourth_s1, encoder_fourth_s2])

        attention = keras.layers.dot([decoder_second, encoder_fourth], axes=[2, 2])
        attention = keras.layers.Activation('softmax')(attention)
        context = keras.layers.dot([attention, encoder_fourth], axes=[2, 1])

        decoder_third = keras.layers.concatenate([context, decoder_second])

        decoder_fourth = keras.layers.LSTM(64, return_sequences=True)(decoder_third)
        decoder_fifth = keras.layers.LSTM(64, return_sequences=True)(decoder_fourth)
        decoder_sixth = keras.layers.LSTM(64, return_sequences=True)(decoder_fifth)

        ##Output Section##
        output = keras.layers.TimeDistributed(keras.layers.Dense(output_train.shape[2]))(decoder_sixth)

        model = keras.Model(inputs=input_train, outputs=output)
        opt = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='mse', optimizer=opt, metrics=['mae'])
        model.summary()

        return model

    else:
        K.clear_session()

        input_train = keras.layers.Input(shape=(shape[1], shape[2]))
        output_train = keras.layers.Input(shape=(shape[4], shape[5]))

        ## Encoder Section##
        encoder_first = keras.layers.LSTM(64, return_sequences=True, return_state=False)(input_train)
        encoder_second = keras.layers.LSTM(64, return_sequences=True)(encoder_first)
        encoder_third = keras.layers.LSTM(64, return_sequences=True)(encoder_second)
        encoder_fourth, encoder_fourth_s1, encoder_fourth_s2 = keras.layers.LSTM(64,return_sequences=False, return_state=True)(encoder_third)

        ##Decorder Section##
        decoder_first = keras.layers.RepeatVector(output_train.shape[1])(encoder_fourth)
        decoder_second = keras.layers.LSTM(64, return_state=False, return_sequences=True)(decoder_first,initial_state=[encoder_fourth,encoder_fourth_s2])
        decoder_third = keras.layers.LSTM(64,return_sequences=True)(decoder_second)
        decoder_fourth = keras.layers.LSTM(64,return_sequences=True)(decoder_third)
        decoder_fifth = keras.layers.LSTM(64,return_sequences=True)(decoder_fourth)
        print(decoder_fifth)

        ##Output Section##
        output = keras.layers.TimeDistributed(keras.layers.Dense(output_train.shape[2]))(decoder_fifth)

        model = keras.Model(inputs=input_train, outputs=output)
        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=opt, metrics=['mae'])
        model.summary()

        return model

def build_mlp(shape):

    K.clear_session()
    model = keras.Sequential()
    model.add(keras.layers.Dense(32, input_shape=(shape[1], shape[2])))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(16))
    model.add(keras.layers.Dense(1))

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])

    model.summary()

    return model

def build_wavenet(shape,attent=False):
    n_filters = 32
    filter_width = 2
    dilation_rates = [2**i for i in range(7)]
   
    inputs = keras.layers.Input(shape=(shape[1],shape[2]))
    x=inputs

    skips = []
    for dilation_rate in dilation_rates:
        
        x   = keras.layers.Conv1D(16, 1, padding='same', activation='relu')(x) 
        x_f = keras.layers.Conv1D(filters=n_filters,kernel_size=filter_width,padding='causal',dilation_rate=dilation_rate)(x)
        x_g = keras.layers.Conv1D(filters=n_filters,kernel_size=filter_width, padding='causal',dilation_rate=dilation_rate)(x)
        
        z = keras.layers.Multiply()([keras.layers.Activation('tanh')(x_f),keras.layers.Activation('sigmoid')(x_g)])
        
        z = keras.layers.Conv1D(16, 1, padding='same', activation='relu')(z)

        x = keras.layers.Add()([x, z])    

        skips.append(z)

    out = keras.layers.Activation('relu')(keras.layers.Add()(skips)) 
    if attent:
        out = attention(return_sequences=True)(out)
    out = keras.layers.Conv1D(128, 1, padding='same')(out)
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.Dropout(0.4)(out)
    out = keras.layers.Conv1D(1, 1, padding='same')(out)
    
    out = keras.layers.Flatten()(out)
    out = keras.layers.Dense(6)(out)
    
    model = keras.Model(inputs, out)
    
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])
    
    return model