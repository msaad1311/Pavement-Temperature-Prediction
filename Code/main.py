import Utils
import Models
import Train
import Ensembler

import pandas as pd
# from IPython.display import display
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import time

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

print('Libaries Loaded')

## Sanity check for GPU
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    
def main():
    
    ##--------------------------------------- Parameters --------------------------------------------##
    PATH = ['C:/Users/Saad.Lakes/Desktop/Lakes-Software-New/Dataset/KuwaitNN_AMS_Data/AMS 01 - AlMutlah_AMS_Data.xlsx']
    
    INPUTS = ['Date','Conc,SO2,2']
    
    INTERMEDIATE = [['Mutlah SO2 Conc']]
    
    OUTPUT = ['Mutlah SO2 Conc']
    
    TARGET_PATH = r'C:\Users\Saad.LAKES\Desktop\Lakes-Software-New\Notebooks\Twenty Four Hour Ahead\Mutlah\SO2 Univariate - Mutlah Only'
    
    DURATION = 24
    
    TRAIN_SIZE = 0.9
    
    SCALE = True
    
    LAG = 24
    
    MODEL=['Wavenet']#'ConvLSTM','CNN-LSTM','Seq2Seq','LSTM'] 
    
    BATCH_SIZE=512
    
    EPOCHS=200
    
    
    ## ------------------------------------------ Main ------------------------------------------------- ##
    assert len(INTERMEDIATE) == len(PATH),'Kindly recheck your INTERMEDIATE Variable'
    
    original_path = os.getcwd()
    target_path = TARGET_PATH.replace('\\','/')
    try:
        os.chdir(target_path)
        print('Directory is present there')
    except FileNotFoundError:
        os.chdir(os.path.split(target_path)[0])
        os.mkdir(os.path.split(target_path)[1])
        os.chdir(target_path)
        print('Crerating new directory')
    finally:
        print('The working directory is moved the Target Path !!!')
    
   
    dfs = [Utils.read_data(p,3,6,INPUTS,o,True) for p,o in zip(PATH,INTERMEDIATE)]
    dfs_flatten = [item for sublist in dfs for item in sublist]
    
    final_df = pd.concat(dfs,axis=1,ignore_index=True)
    final_df.columns=dfs_flatten

    # display(final_df.head())
       
    x_train,x_test,y_train,y_test,scaler = Utils.splitter(final_df,OUTPUT,LAG,DURATION,TRAIN_SIZE,True)
    
    ###############################################
#     Train.build_model('ConvLSTM',final_df,OUTPUT[0],scaler,x_train,x_test,y_train,y_test,BATCH_SIZE,EPOCHS)
#     time.sleep(2)
#     print('-'*30 + 'The program is restarting'+'-'*30)
#     K.clear_session()
#     tf.compat.v1.reset_default_graph()
#     ###############################################
#     Train.build_model('CNN-LSTM',final_df,OUTPUT[0],scaler,x_train,x_test,y_train,y_test,BATCH_SIZE,EPOCHS)
#     time.sleep(2)
#     print('-'*30 + 'The program is restarting'+'-'*30)
#     K.clear_session()
#     tf.compat.v1.reset_default_graph()
#     ###############################################
#     Train.build_model('Seq2Seq',final_df,OUTPUT[0],scaler,x_train,x_test,y_train,y_test,BATCH_SIZE,EPOCHS)
#     time.sleep(2)
#     print('-'*30 + 'The program is restarting'+'-'*30)
#     K.clear_session()
#     tf.compat.v1.reset_default_graph()
#     ################################################
#     Train.build_model('LSTM',final_df,OUTPUT[0],scaler,x_train,x_test,y_train,y_test,BATCH_SIZE,EPOCHS)
#     time.sleep(2)
#     print('-'*30 + 'The program is restarting'+'-'*30)
#     K.clear_session()
#     tf.compat.v1.reset_default_graph()
    ###############################################
    Train.build_model('Wavenet',final_df,OUTPUT[0],scaler,x_train,x_test,y_train,y_test,BATCH_SIZE,EPOCHS)
    time.sleep(2)
    print('-'*30 + 'The program is restarting'+'-'*30)
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    
#     Ensembler.ensembler(MODEL)
    
    os.chdir(original_path)
    
    print('The working is moved again to the original path!!!')
    
if __name__ == "__main__":
    main()

# 0.0990596661118757 -> trial one
# 