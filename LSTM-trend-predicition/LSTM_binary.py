import pandas as pd
import numpy as np
from keras import Sequential, optimizers, regularizers
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.models import load_model, save_model
import matplotlib.pyplot as plt

''' Load and put data into time series format
'''
raw_data = pd.read_csv('./data/preprocessed_6hr_data_lastYear_standardized_BB.csv')
data = raw_data.values # convert to numpy arrays
Y_raw = raw_data['Close'].values # The Close values will be used as labels

print(data.shape)

# Data size. Building the training data tensor X_construct will be shape (m, n_f, t)
timesteps = 1 # Number of timesteps to treat as input feature
m = data.shape[0] - timesteps
n_f = data.shape[1]

''' The data is currently in dimension (n_examples, n_features).
We want to add a third dimension, for timesteps. Each new timestep entry will be the same data, shifted by
a timestep. For example, if there are 3 timesteps, the the third dimension will be size 3 where the first matrix
is the original data, the second matrix is the original data shifted up by one row, and the third matrix is the original
data shifted up by two rows. This would repeat t timesteps. 
The data will end in final shape of (m, n_f, t)
'''

X_construct = data[:-timesteps] # t timesteps must be shaved off the bottom of the data
X_construct = X_construct.reshape(m, n_f, 1) # reshape with a 3rd dimension for t
for t in range(1, timesteps):
    X_ti = np.copy(data[t:m+t]).reshape(m, n_f, 1) # original data shifted by t rows
    X_construct = np.concatenate((X_construct, X_ti), axis=2) # add this timesteps matrix to the tensor

print(X_construct.shape)

''' For Y, the labels, we want 1s and 0s to represent trend up vs trend down.
Y_change calculates the change in price, which is converted to 1s and 0s using numpy'''
Y_diff = raw_data['Close'].diff(periods=1).iloc[timesteps:].values # Get the difference in price
Y = np.asarray((Y_diff > 0)).astype(float) # Convert to 1 (trend up), or 0 (trend down)
print(Y.shape)

X = X_construct.swapaxes(1,2) # shape (m, n_f, t) -> (m, t, n_f) for keras LSTM
Y = Y.reshape(len(Y), 1) # Column vector shape
print(X.shape)
print(Y.shape)

# Split training and testing data
test_split = int(0.9*len(Y))
X_train, X_test = X[:test_split], X[test_split:]
Y_train, Y_test = Y[:test_split], Y[test_split:]

def build_model(timesteps, data_dim):
    model = Sequential()

    model.add(LSTM(128, activation='tanh', kernel_initializer='he_uniform',
                   return_sequences=True, input_shape=(timesteps, data_dim)))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))


    model.add(LSTM(64, activation='tanh', kernel_initializer='he_uniform'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model

loading = False
saving = True

if loading:
    model = load_model('./models/model_name.h5')
else:
    model = build_model(timesteps, data_dim=n_f)

epochs = 120


history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=128, epochs=epochs, shuffle=False, verbose=2)
model.save('./models/LSTM-6hr-standardized.h5')
print(model.summary())

''' Plot Results'''
### Plot results.
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label='val_acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
# summarize history for loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()