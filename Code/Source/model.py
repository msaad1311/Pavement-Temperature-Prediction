import os
import matplotlib.pylab as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error as mse
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers.normalization import BatchNormalization


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


def model_fit(dest, directory, model, xtrain, ytrain, xtest, ytest, scaler, checkpoint, weights, sheet):
    old_path = os.getcwd()
    try:
        os.chdir(os.path.join(dest, directory))
        print('Directory present')
    except FileNotFoundError:
        print('Creating a new directory......')
        os.chdir(os.path.join(dest))
        os.mkdir(directory)
        os.chdir(os.path.join(dest, directory))
        print('New Directory Created')

    history = model.fit(xtrain, ytrain, validation_split=0.1,
                        batch_size=32, epochs=200, callbacks=[checkpoint])

    plt.plot(history.history['loss'], 'r', label='Training Loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.legend()
    plt.show()

    model.load_weights(weights)
    preds = model.predict(xtest)

    ytest_unscaled = scaler.inverse_transform(ytest)
    preds_unscaled = scaler.inverse_transform(preds)

    for i in range(ytest.shape[1]):
        sheet.write(0, 0, 'MSE')
        sheet.write(0, 1, 'Hours Ahead')
        sheet.write(i + 1, 0, mse(ytest_unscaled[:, i], preds_unscaled[:, i]))
        sheet.write(i + 1, 1, i+1)
    os.chdir(old_path)


class buildLSTM(BaseEstimator, TransformerMixin):
    def __init__(self, X, atten,checkpoint):
        self.X = X
        self.atten = atten
        self.checkpoint = checkpoint
        self.model = keras.Sequential()
        self.model.add(keras.layers.LSTM(64, return_sequences=True,
                       input_shape=(self.X.shape[1], self.X.shape[2])))
        self.model.add(keras.layers.LSTM(64, return_sequences=True))
        self.model.add(keras.layers.Dropout(0.3))
        self.model.add(keras.layers.LSTM(64, return_sequences=True))
        self.model.add(keras.layers.LSTM(64, return_sequences=True))
        if self.atten:
            self.model.add(attention(return_sequences=True))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(512, activation='relu'))
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dense(64, activation='relu'))
        self.model.add(keras.layers.Dropout(0.3))
        self.model.add(keras.layers.Dense(32))
        self.model.add(keras.layers.Dense(6))

        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(
            learning_rate=3e-4), metrics=['mae'])

    @staticmethod
    def checkpointer(atten):
        if atten:
            filepath = 'attention_lstm.hdf5'
        else:
            filepath = 'simple_lstm.hdf5'
        checkpoint = keras.callbacks.ModelCheckpoint(filepath,monitor='val_loss',save_best_only=True)
        return checkpoint

    def fit(self, X, y):
        checkpoint = self.checkpointer(self.atten)
        self.history = self.model.fit(X,y,validation_split=0.1,
                        batch_size=32, epochs=200, callbacks=[checkpoint])
        plt.plot(self.history.history['loss'], 'r', label='Training Loss')
        plt.plot(self.history.history['val_loss'], 'b', label='Validation Loss')
        plt.legend()
        plt.show()
        
        return self.model
    
    def predict(self,model,weights,scaler,xtest,ytest):

        self.model.load_weights(weights)
        preds = self.model.predict(xtest)

        ytest_unscaled = scaler.inverse_transform(ytest)
        preds_unscaled = scaler.inverse_transform(preds)
        


def build_lstm(x_train, atten=False, dropout=False, regularizer=None):
    K.clear_session()
    model = keras.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LSTM(64, return_sequences=True,
              kernel_regularizer=regularizer, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LSTM(
        64, kernel_regularizer=regularizer, return_sequences=True))
    if dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LSTM(
        64, kernel_regularizer=regularizer, return_sequences=True))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LSTM(
        64, kernel_regularizer=regularizer, return_sequences=True))
    if atten:
        model.add(attention(return_sequences=True))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    if dropout:
        model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dense(6))

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(
        learning_rate=0.001), metrics=['mae'])

    return model


def build_cnnlstm(x_train, atten=False, regularizer=None):
    model = keras.Sequential()
    model.add(keras.layers.add(BatchNormalization()))
    model.add(keras.layers.Conv1D(64, kernel_size=3, kernel_regularizer=regularizer,
              input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(keras.layers.add(BatchNormalization()))
    model.add(keras.layers.Conv1D(64, kernel_size=3))
    model.add(keras.layers.add(BatchNormalization()))
    model.add(keras.layers.LSTM(64, return_sequences=True))
    model.add(keras.layers.add(BatchNormalization()))
    model.add(keras.layers.LSTM(64, return_sequences=True))
    if atten:
        model.add(attention(return_sequences=True))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dense(6))

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(
        learning_rate=0.0001), metrics=['mae'])
    return model
