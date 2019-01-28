import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, SimpleRNN, LSTM
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from pandas.core.frame import DataFrame
from datetime import datetime

def load_dataset():
    ds = pd.read_csv('hour.csv')
    ds['dteday'] = pd.to_datetime(ds['dteday'])
    return ds

def filter_by_date(ds: DataFrame, start_date: str, end_date: str):
    
    start_date_parsed = datetime.strptime(start_date, "%Y-%m-%d")
    start_end_parsed = datetime.strptime(end_date, "%Y-%m-%d")
    
    return ds[(ds['dteday'] >= start_date_parsed) & (ds['dteday'] <= start_end_parsed)]

def preprocess_dataset(ds):
    ds_reduced = ds[['yr','mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]
    return preprocessing.scale(ds_reduced.values), ds['cnt'].values

def get_model():

    input = Input(shape=(1, 11))
    rnn = LSTM(50)(input)
    activation = Dense(1, activation='linear')(rnn)

    model = Model(inputs=input, outputs=activation)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    return model

def train_model(model, X_train, Y_train):
    X = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    Y = Y_train
    model.fit(X, Y, epochs=5000, batch_size=128, verbose=1)
    return model

dataset = load_dataset()
train = filter_by_date(dataset, '2011-01-01', '2012-11-30')
validation = filter_by_date(dataset, '2011-11-01', '2012-11-30')

X_train, Y_train = preprocess_dataset(train)
X_validation, Y_validation = preprocess_dataset(validation)

model = train_model(get_model(), X_train, Y_train)

def validate_model(model, X_validation, Y_validation):
    X = np.reshape(X_validation, (X_validation.shape[0], 1, X_validation.shape[1]))
    Y = model.predict(X)
    return mean_squared_error(Y_validation, Y)

print(validate_model(model, X_validation, Y_validation))
