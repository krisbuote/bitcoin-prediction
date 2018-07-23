from keras import models
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

''' Provide location for data'''
data_raw = pd.read_pickle('C:/Users/Admin/PycharmProjects/BTC-data/bitcoin-historical-data-kaggle/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.pkl')
data_raw = data_raw.values # Turn into np array


'''Build the net with Keras'''
def model(neurons, timesteps, data_dim):
    model = models.Sequential()
    # model.add(TimeDistributed(Dense(128), input_shape=(timesteps,data_dim)))
    # model.add(Dropout(0.8))
    # model.add(LSTM(neurons, return_sequences=True))
    # model.add(Dropout(0.5))
    model.add(LSTM(neurons, return_sequences=False,input_shape=(timesteps,data_dim)))
    model.add(Dense(5)) #n neurons for outputting a n price predictions
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


'''Data prep'''
#Reshape Timeseries data from 1 column to n=timesteps columns, each shifted by n
#For example, if timesteps = 3, this will produce numpy arrays of 3 timesteps each used for training.
def timeseries_to_supervised(data, timesteps):
    df = pd.DataFrame(data)
    columns = [df.shift(-i) for i in range(0, timesteps)]
    df.append(columns)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    df = df.drop(df.index[-timesteps:])
    return df.values

# scale train and test data to [-1, 1]
def scale(raw_data, split_pct):
    train_test_split_index = int(split_pct * len(raw_data))  # Use (int) percent of data to train
    train = raw_data[:train_test_split_index]  # split the timestep shifted data for x
    # fit scaler on only training data but apply to everything.
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    raw_data_scaled = scaler.transform(raw_data)
    return scaler, raw_data_scaled

#Training data should be in shape (examples, timesteps, data_dim)
# Test data should be in shape (examples, data_dim)
def split_and_reshape(data, data_raw, split_pct, timesteps, data_dim):
    train_test_split_index = int(split_pct * len(data))  # Use (int) percent of data to train
    x_train = data[:train_test_split_index]  # split the timestep shifted data for x
    x_test = data[train_test_split_index:]
    y_train = data_raw[timesteps:train_test_split_index + timesteps]  # split the original data for labels
    y_test = data_raw[train_test_split_index + timesteps:]

    x_train = x_train.reshape(x_train.shape[0], timesteps, data_dim)
    y_train = y_train.reshape(y_train.shape[0], data_dim)
    x_test = x_test.reshape(x_test.shape[0], timesteps, data_dim)
    y_test = y_test.reshape(y_test.shape[0], data_dim)

    return x_train, y_train, x_test, y_test
