import os
import pandas as pd
import matplotlib.pylab as plt

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

class buildLSTM():
    def __init__(self, xtrain, ytrain, scaler, atten):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.scaler = scaler
        self.atten = atten
        self.model = self.buildModel(self.xtrain, self.atten)
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

    def fit(self):
        checkpoint = self.checkpointer(self.atten)
        self.history = self.model.fit(self.xtrain, self.ytrain, validation_split=0.1,
                                      batch_size=32, epochs=2, callbacks=[checkpoint])
        plt.plot(self.history.history['loss'], 'r', label='Training Loss')
        plt.plot(self.history.history['val_loss'],
                 'b', label='Validation Loss')
        plt.legend()
        plt.show()

        return self

    def predictions(self, xtest, ytest):
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
            results.append(mse(ytest_unscaled[:, i], preds_unscaled[:, i]))
        results = pd.DataFrame(results).reset_index()
        path = f'../Results/LSTM/MSE_{self.atten}.xlsx'
        try:
            results.to_excel(path)
        except Exception as e:
            print(e)
            os.mkdir(os.path.split(path)[0])
            results.to_excel(path)
        return self

class buildCNNLSTM():
    def __init__(self, xtrain, ytrain, scaler, atten):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.scaler = scaler
        self.atten = atten
        self.model = self.buildModel(self.xtrain, self.atten)
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
        model.add(keras.layers.Conv1D(64, kernel_size=3,
                  input_shape=(x.shape[1], x.shape[2])))
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

    def predictions(self, xtest, ytest):
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
            results.append(mse(ytest_unscaled[:, i], preds_unscaled[:, i]))
        results = pd.DataFrame(results).reset_index()
        path = f'../Results/CNNLSTM/MSE_{self.atten}.xlsx'
        try:
            results.to_excel(path)
        except Exception as e:
            print(e)
            os.mkdir(os.path.split(path)[0])
            results.to_excel(path)
        return self

class buildConvLSTM():
    def __init__(self, xtrain, ytrain, scaler, atten):
        self.xtrain = self.shapeSetter(xtrain)
        self.ytrain = ytrain
        self.scaler = scaler
        self.atten = atten
        self.model = self.buildModel(self.xtrain, self.atten)
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=3e-4), metrics=['mae'])

    @staticmethod
    def shapeSetter(x):
        return x.reshape(x.shape[0], 1, 1, x.shape[1], x.shape[2])

    @staticmethod
    def buildModel(x, atten):
        model = keras.Sequential()
        model.add(keras.layers.ConvLSTM2D(64, kernel_size=(1, 2), return_sequences=True,
                                          input_shape=(x.shape[1], x.shape[2],
                                                       x.shape[3], x.shape[4])))
        model.add(keras.layers.ConvLSTM2D(64, kernel_size=(1, 2), return_sequences=True))
        model.add(keras.layers.ConvLSTM2D(64, kernel_size=(1, 2), return_sequences=True))
        model.add(keras.layers.ConvLSTM2D(64, kernel_size=(1, 2), return_sequences=True))
        if atten:
            model.add(attention(return_sequences=True))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(32))
        model.add(keras.layers.Dense(6))
        
        return model
    @staticmethod
    def checkpointer(atten):
        if atten:
            filepath = '../Weights/ConvLSTM/attention_convlstm.hdf5'
        else:
            filepath = '../Weights/ConvLSTM/simple_convlstm.hdf5'
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath, monitor='val_loss', save_best_only=True)
        return checkpoint
    
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

    def predictions(self, xtest, ytest):
        if self.atten:
            filepath = '../Weights/ConvLSTM/attention_convlstm.hdf5'
        else:
            filepath = '../Weights/ConvLSTM/simple_convlstm.hdf5'
        print(self.model)
        self.model.load_weights(filepath)
        preds = self.model.predict(self.shapeSetter(xtest))

        ytest_unscaled = self.scaler.inverse_transform(ytest)
        preds_unscaled = self.scaler.inverse_transform(preds)

        results = []
        for i in range(ytest.shape[1]):
            results.append(mse(ytest_unscaled[:, i], preds_unscaled[:, i]))
        results = pd.DataFrame(results).reset_index()
        path = f'../Results/ConvLSTM/MSE_{self.atten}.xlsx'
        try:
            results.to_excel(path)
        except Exception as e:
            print(e)
            os.mkdir(os.path.split(path)[0])
            results.to_excel(path)
        return self

class buildSeq2Seq():
    def __init__(self,xtrain,ytrain,scaler,atten):
        self.xtrain = xtrain
        self.ytrain = self.shapeSetter(ytrain)
        self.scaler = scaler
        self.atten  = atten
        self.model  = self.buildModel(self.xtrain,self.ytrain,self.atten)
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=3e-4), metrics=['mae'])
        
    @staticmethod
    def buildModel(xtrain,ytrain,atten):
        if not atten:
            input_train = keras.layers.Input(shape=(xtrain.shape[1], xtrain.shape[2]))
            output_train = keras.layers.Input(shape=(ytrain.shape[1], ytrain.shape[2]))

            ### --------------------------------Encoder Section -----------------------------------------------###
            encoder_first = keras.layers.LSTM(128, return_sequences=True, return_state=False)(input_train)
            encoder_second = keras.layers.LSTM(128, return_sequences=True)(encoder_first)
            encoder_third = keras.layers.LSTM(128, return_sequences=True)(encoder_second)
            encoder_fourth, encoder_fourth_s1, encoder_fourth_s2 = keras.layers.LSTM(128,return_sequences=False, return_state=True)(encoder_third)

            ###---------------------------------Decorder Section-----------------------------------------------###
            decoder_first = keras.layers.RepeatVector(output_train.shape[1])(encoder_fourth)
            decoder_second = keras.layers.LSTM(128, return_state=False, return_sequences=True)(decoder_first,initial_state=[encoder_fourth,encoder_fourth_s2])
            decoder_third = keras.layers.LSTM(128,return_sequences=True)(decoder_second)
            decoder_fourth = keras.layers.LSTM(128,return_sequences=True)(decoder_third)
            decoder_fifth = keras.layers.LSTM(128,return_sequences=True)(decoder_fourth)
            print(decoder_fifth)

            ###--------------------------------Output Section-------------------------------------------------###
            output = keras.layers.TimeDistributed(keras.layers.Dense(output_train.shape[2]))(decoder_fifth)

            model = keras.Model(inputs=input_train, outputs=output)
            
            return model
        else:
            input_train = keras.layers.Input(shape=(xtrain.shape[1], xtrain.shape[2]))
            output_train = keras.layers.Input(shape=(ytrain.shape[1], ytrain.shape[2]))

            ###----------------------------------------Encoder Section------------------------------------------###
            encoder_first = keras.layers.LSTM(128, return_sequences=True, return_state=False)(input_train)
            encoder_second = keras.layers.LSTM(128, return_sequences=True)(encoder_first)
            encoder_third = keras.layers.LSTM(128, return_sequences=True)(encoder_second)
            encoder_fourth, encoder_fourth_s1, encoder_fourth_s2 = keras.layers.LSTM(128,return_sequences=True,return_state=True)(encoder_third)

            ###-----------------------------------------Decoder Section------------------------------------------###
            decoder_first = keras.layers.RepeatVector(output_train.shape[1])(encoder_fourth_s1)
            decoder_second = keras.layers.LSTM(128, return_state=False, return_sequences=True)(decoder_first, initial_state=[encoder_fourth_s1, encoder_fourth_s2])

            attention = keras.layers.dot([decoder_second, encoder_fourth], axes=[2, 2])
            attention = keras.layers.Activation('softmax')(attention)
            context = keras.layers.dot([attention, encoder_fourth], axes=[2, 1])

            decoder_third = keras.layers.concatenate([context, decoder_second])

            decoder_fourth = keras.layers.LSTM(128, return_sequences=True)(decoder_third)
            decoder_fifth = keras.layers.LSTM(128, return_sequences=True)(decoder_fourth)
            decoder_sixth = keras.layers.LSTM(128, return_sequences=True)(decoder_fifth)

            ###-----------------------------------------Output Section-----------------------------------------###
            output = keras.layers.TimeDistributed(keras.layers.Dense(output_train.shape[2]))(decoder_sixth)

            model = keras.Model(inputs=input_train, outputs=output)
            return model
        
    @staticmethod
    def shapeSetter(X):
        return X.reshape(X.shape[0], X.shape[1], 1)
    
    @staticmethod
    def checkpointer(atten):
        if atten:
            filepath = '../Weights/Seq2Seq/attention_seq2seq.hdf5'
        else:
            filepath = '../Weights/Seq2Seq/simple_seq2seq.hdf5'
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath, monitor='val_loss', save_best_only=True)
        return checkpoint
    
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
    
    def predictions(self, xtest, ytest):
        if self.atten:
            filepath = '../Weights/Seq2Seq/attention_seq2seq.hdf5'
        else:
            filepath = '../Weights/Seq2Seq/simple_seq2seq.hdf5'
        print(self.model)
        self.model.load_weights(filepath)
        preds = self.model.predict(xtest)
        preds = preds.reshape(preds.shape[0],preds.shape[1])
        
        ytest_unscaled = self.scaler.inverse_transform(ytest)
        preds_unscaled = self.scaler.inverse_transform(preds)

        results = []
        for i in range(ytest.shape[1]):
            results.append(mse(ytest_unscaled[:, i], preds_unscaled[:, i]))
        results = pd.DataFrame(results).reset_index()
        path = f'../Results/Seq2Seq/MSE_{self.atten}.xlsx'
        try:
            results.to_excel(path)
        except Exception as e:
            print(e)
            os.mkdir(os.path.split(path)[0])
            results.to_excel(path)
        return self
            
class buildWavenet():
    def __init__(self,xtrain,ytrain,scaler,atten,n_filters=128,filter_width=2) -> None:
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.dilation_rates = [2**i for i in range(7)]
        self.atten = atten
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.scaler = scaler
        self.model = self.buildModel(self.n_filters,self.filter_width,self.dilation_rates,self.xtrain,self.atten)
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=3e-4), metrics=['mae'])
    
    @staticmethod
    def buildModel(n_filters,filter_width,dilation_rates,xtrain,atten):
        inputs = keras.layers.Input(shape=(xtrain.shape[1],xtrain.shape[2]))
        x=inputs

        skips = []
        for dilation_rate in dilation_rates:

            x   = keras.layers.Conv1D(64, 1, padding='same')(x) 
            x_f = keras.layers.Conv1D(filters=n_filters,kernel_size=filter_width,padding='causal',dilation_rate=dilation_rate)(x)
            x_g = keras.layers.Conv1D(filters=n_filters,kernel_size=filter_width, padding='causal',dilation_rate=dilation_rate)(x)

            z = keras.layers.Multiply()([keras.layers.Activation('tanh')(x_f),keras.layers.Activation('sigmoid')(x_g)])

            z = keras.layers.Conv1D(64, 1, padding='same', activation='relu')(z)

            x = keras.layers.Add()([x, z])    

            skips.append(z)

        out = keras.layers.Activation('relu')(keras.layers.Add()(skips)) 
        if atten:
            out = attention(return_sequences=True)(out)
        out = keras.layers.Conv1D(128, 1, padding='same')(out)
        out = keras.layers.Activation('relu')(out)
        out = keras.layers.Dropout(0.4)(out)
        out = keras.layers.Conv1D(1, 1, padding='same')(out)

        out = keras.layers.Flatten()(out)
        out = keras.layers.Dense(6)(out)

        model = keras.Model(inputs, out)
        return model
    
    @staticmethod
    def checkpointer(atten):
        if atten:
            filepath = '../Weights/WaveNet/attention_wavenet.hdf5'
        else:
            filepath = '../Weights/WaveNet/simple_wavenet.hdf5'
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath, monitor='val_loss', save_best_only=True)
        return checkpoint
    
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
    
    def predictions(self, xtest, ytest):
        if self.atten:
            filepath = '../Weights/WaveNet/attention_wavenet.hdf5'
        else:
            filepath = '../Weights/WaveNet/simple_wavenet.hdf5'
            
        self.model.load_weights(filepath)
        preds = self.model.predict(xtest)
        preds = preds.reshape(preds.shape[0],preds.shape[1])
        
        ytest_unscaled = self.scaler.inverse_transform(ytest)
        preds_unscaled = self.scaler.inverse_transform(preds)

        results = []
        for i in range(ytest.shape[1]):
            results.append(mse(ytest_unscaled[:, i], preds_unscaled[:, i]))
        results = pd.DataFrame(results).reset_index()
        path = f'../Results/Wavenet/MSE_{self.atten}.xlsx'
        try:
            results.to_excel(path)
        except Exception as e:
            print(e)
            os.mkdir(os.path.split(path)[0])
            results.to_excel(path)
        return self
    

        
