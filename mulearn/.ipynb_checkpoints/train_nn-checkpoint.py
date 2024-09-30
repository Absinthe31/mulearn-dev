import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import random
from fknn import FuzzyKNN
from sklearn.svm import SVC, SVR
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import keras_tuner
import logging
from logging import FileHandler
#import cloudpickle


def gaussian(x, var):
    y = (np.e ** (-(((x-0.5)*2)**2)/(2*var)))
    return y

X_test = np.linspace(0, 1, 1000)
#X_test = np.array([x for x in X_test if (gaussian(x, np.var(X_test)) <= 0.1 or gaussian(x, np.var(X_test)) >= 0.9)])
#X_test2 = [np.array([x for x in X_test if x<0.2]),np.array([x for x in X_test if (x>=0.2 and x<0.6)]),np.array([x for x in X_test if x>0.8])]

dataset = pd.read_csv("synthetic_dataset.csv")

X_tr, X_ts, Y_tr, Y_ts = train_test_split(dataset.X, dataset.y, test_size=0.2, random_state=42)


def build_model(hp):
    model = Sequential()
    model.add(Dense(1, input_shape=(1,), activation='relu'))
    model.add(Dense(
        hp.Choice('units', [10, 15, 20, 25, 30]),
        activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(
        hp.Choice('units', [10, 15, 20, 25, 30]),
        activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(
        hp.Choice('units', [10, 15, 20, 25, 30]),
        activation='relu'))
    model.add(Dropout(0.2)) 
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model

tuner = keras_tuner.GridSearch(
    build_model,
    objective='val_loss',
    max_trials=5)

tuner.search(X_tr, Y_tr, epochs=100, validation_data=(X_ts, Y_ts))

best_model = tuner.get_best_models()[0]

model = best_model

model.summary()

batch_size = 16
epochs = 400


class PlotTraining(tf.keras.callbacks.Callback):
    def __init__(self):
        pass

    def on_train_begin(self, logs={}):
        plt.ion()

    def on_epoch_end(self, epoch, logs={}):
        pass

class JsonFormatter(logging.Formatter):

    @staticmethod
    def find(s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    def format(self, record) -> str:
        
        log_record = {
            'timestamp': self.formatTime(record),
            'model': record.getMessage()
        }
        
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)

        output = json.dumps(log_record) + ','

        output = output[:self.find(output, '"')[-1]] + '' + output[self.find(output, '"')[-1]+1:]
        output = output[:self.find(output, '"')[-1]] + '' + output[self.find(output, '"')[-1]+1:]

        output = output.replace("'",'"')
        
        return output

def log_model(model, score=-1, acc=-1, model_name='', serialize=False, dataset=''):

    json_dict = { 'model_name' : model.__repr__() if model_name == '' else model_name }

    if score != -1:
        json_dict['score'] = score
    if acc != -1:
        json_dict['acc'] = acc
    if serialize == True:
        json_dict['obj'] = cloudpickle.dumps(model)      
    if dataset != '':
        json_dict['dataset'] = dataset

    return json_dict

json_handler = FileHandler('test_logs.log')
json_handler.setLevel(logging.INFO)
json_formatter = JsonFormatter()
json_handler.setFormatter(json_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(json_handler)
logging.getLogger().setLevel(logging.INFO)


model.compile(loss='mean_squared_error', optimizer='sgd')

model.fit(X_tr,
          Y_tr.astype(np.float64),
          epochs=epochs,
          batch_size=batch_size, 
          verbose=1, 
          validation_data=(X_ts, Y_ts), 
          callbacks=[tf.keras.callbacks.History()])

plt.plot(X_test, model.predict(X_test.reshape(-1,1)), color="blue", label="learnt f.")
#plt.plot(X_test2[1], model.predict(X_test2[1].reshape(-1,1)), color="blue")
#plt.plot(X_test2[2], model.predict(X_test2[2].reshape(-1,1)), color="blue")

plt.plot(X_test, gaussian(X_test, np.var(X_test)), color="red", label="original f.")
#plt.plot(X_test2[1], gaussian(X_test, np.var(X_test)), color="red")
#plt.plot(X_test2[2], gaussian(X_test, np.var(X_test)), color="red")

plt.legend(loc="upper right")

#step = (X_test2[0][-1]-X_test2[0][0])/(len(X_test2[0])-1)

step = (X_test[-1]-X_test[0])/(len(X_test)-1)

err = np.sum(step*np.abs(model.predict(X_test.reshape(-1,1)) - gaussian(X_test, np.var(X_test))))

print("err: ", err)

plt.show()



