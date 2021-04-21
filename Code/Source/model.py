import os
import pandas as pd
import matplotlib.pylab as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error as mse

from tensorflow import keras
import tensorflow.keras.backend as K


class attention(keras.layers.Layer):
    '''
    Attention layer for the neural networks.

    if return_sequences=True, it will give 3D vector and if false it will give 2D vector. It is same as LSTMs.

    https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137
    the  following code is being inspired from the above link.
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

class buildLSTM(BaseEstimator, TransformerMixin):
    def __init__(self,xtrain, scaler,atten):
        self.xtrain = xtrain
        self.scaler = scaler
        self.atten = atten
        self.model = self.buildModel(self.xtrain,self.atten)
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(
            learning_rate=3e-4), metrics=['mae'])

    @staticmethod
    def checkpointer(atten):
        if atten:
            filepath = '../Weights/LSTM/attention_lstm.hdf5'
        else:
            filepath = '../Weights/LSTM/simple_lstm.hdf5'
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath, monitor='val_loss', save_best_only=True)
        return checkpoint

    @staticmethod
    def buildModel(x, atten):
        model = keras.Sequential()
        model.add(keras.layers.LSTM(64, return_sequences=True,
                  input_shape=(x.shape[1], x.shape[2])))
        model.add(keras.layers.LSTM(64, return_sequences=True))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.LSTM(64, return_sequences=True))
        model.add(keras.layers.LSTM(64, return_sequences=True))
        if atten:
            model.add(attention(return_sequences=True))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(32))
        model.add(keras.layers.Dense(6))

        return model

    def fit(self,xtrain,ytrain):
        checkpoint = self.checkpointer(self.atten)
        self.history = self.model.fit(xtrain, ytrain, validation_split=0.1,
                                      batch_size=32, epochs=2, callbacks=[checkpoint])
        plt.plot(self.history.history['loss'], 'r', label='Training Loss')
        plt.plot(self.history.history['val_loss'],
                 'b', label='Validation Loss')
        plt.legend()
        plt.show()

        return self

    def transform(self,xtest,ytest):
        if self.atten:
            filepath = '../Weights/LSTM/attention_lstm.hdf5'
        else:
            filepath = '../Weights/LSTM/simple_lstm.hdf5'
        self.model.load_weights(filepath)
        preds = self.model.predict(xtest)

        ytest_unscaled = self.scaler.inverse_transform(ytest)
        preds_unscaled = self.scaler.inverse_transform(preds)
        
        results = []
        for i in range(ytest.shape[1]):
            results.append(mse(ytest_unscaled[:,i],preds_unscaled[:,i]))
        results = pd.DataFrame(results).reset_index()
        path = f'../Results/LSTM/MSE_{self.atten}.xlsx'
        try:
            results.to_excel(path)
        except Exception as e:
            print(e)
            os.mkdir(os.path.split(path)[0])
            results.to_excel(path)
        return self
    
class buildCNNLSTM(BaseEstimator, TransformerMixin):
    def __init__(self, xtrain,ytrain,atten):
        self.xtrain = xtrain
        self.ytrain = ytrain
        # self.scaler = scaler
        self.atten = atten
        self.model = self.buildModel(self.xtrain,self.atten)
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(
            learning_rate=3e-4), metrics=['mae'])

    @staticmethod
    def checkpointer(atten):
        if atten:
            filepath = '../Weights/CNNLSTM/attention_cnnlstm.hdf5'
        else:
            filepath = '../Weights/CNNLSTM/simple_cnnlstm.hdf5'
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath, monitor='val_loss', save_best_only=True)
        return checkpoint

    @staticmethod
    def buildModel(x, atten):
        model = keras.Sequential()
        model.add(keras.layers.Conv1D(64, kernel_size=3, input_shape=(x.shape[1],x.shape[2])))
        model.add(keras.layers.Conv1D(64, kernel_size=3))
        model.add(keras.layers.LSTM(64, return_sequences=True))
        model.add(keras.layers.LSTM(64, return_sequences=True))
        if atten:
            model.add(attention(return_sequences=True))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(32))
        model.add(keras.layers.Dense(6))

        return model

    def fit(self):
        checkpoint = self.checkpointer(self.atten)
        self.history = self.model.fit(self.xtrain, self.ytrain, validation_split=0.1,
                                      batch_size=32, epochs=2, callbacks=[checkpoint])
        plt.plot(self.history.history['loss'], 'r', label='Training Loss')
        plt.plot(self.history.history['val_loss'],
                 'b', label='Validation Loss')
        plt.legend()
        plt.show()

        return self.model

    def transform(self,xtest,ytest):
        if self.atten:
            filepath = '../Weights/CNNLSTM/attention_cnnlstm.hdf5'
        else:
            filepath = '../Weights/CNNLSTM/simple_cnnlstm.hdf5'
        print(self.model)
        self.model.load_weights(filepath)
        preds = self.model.predict(xtest)

        ytest_unscaled = self.scaler.inverse_transform(ytest)
        preds_unscaled = self.scaler.inverse_transform(preds)
        
        results = []
        for i in range(ytest.shape[1]):
            results.append(mse(ytest_unscaled[:,i],preds_unscaled[:,i]))
        results = pd.DataFrame(results).reset_index()
        path = f'../Results/CNNLSTM/MSE_{self.atten}.xlsx'
        try:
            results.to_excel(path)
        except Exception as e:
            print(e)
            os.mkdir(os.path.split(path)[0])
            results.to_excel(path)
        return self
