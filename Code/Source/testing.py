import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xlwt 
from xlwt import Workbook 
import utils as u
import model as m

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse

import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

print('Libraries Loaded')

## Specifying the source path
src = r'D:\Pavement-Temperature-Prediction\Data\Pave_data_cleaned.xlsx'
# filename = r'Pave_data_cleaned.xlsx'

## Specifying the destination path
dest = r'D:\Pavement-Temperature-Prediction\Results\Six Hours Lag'

## Reading the file
df = u.read_file(src)

## Creating the training and testing data
x_train,x_test,y_train,y_test,scaler = u.splitter(df[['Temp','Pavement']],['Pavement'],6,6,0.8)
print(f'The shape of x_train is {x_train.shape} and x_test is {x_test.shape}')
print(f'The shape of y_train is {y_train.shape} and y_test is {y_test.shape}')

model = m.buildLSTM(x_train,True)
model = model.fit(x_train,y_train)
model.predict(scaler,x_test,y_test)