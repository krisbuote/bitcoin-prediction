from lstm import model,scale,timeseries_to_supervised,split_and_reshape,data_raw
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy
from sklearn.metrics import mean_squared_error
from math import sqrt

### Parameters
epochs = 1
batch_size = 4
timesteps = 8 #how many timesteps RNN includes
neurons = 1 # Number of neurons in the LSTM
data_dim = data_raw.shape[1] # n_cols in data: only Bitcoin price currently
split_pct = 0.90 #percent of data in training
loading = False #Set to True if loading a saved model
saving = True #Set to True if you wish to save new model

### Preprocess Data
scaler, raw_data_scaled = scale(data_raw, split_pct) #Scale all of the data
data_scaled = timeseries_to_supervised(raw_data_scaled, timesteps) #turn it into timeshifted data
x_train, y_train, x_test, y_test = split_and_reshape(data_scaled, raw_data_scaled, split_pct, timesteps, data_dim) #Uses shifted data for x and original data for y

### Load previous Model or Fit new one
if loading == True:
    model = load_model('./model/my_model.h5')

else:
    model = model(neurons, timesteps, data_dim)
    model_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=False)

if saving == True:
    model.save('./model/my_model.h5')


predictions = model.predict(x_test)
evaluate = model.evaluate(x_test,y_test,batch_size=batch_size)
print("Test Loss is ", evaluate[0])
rmse = sqrt(mean_squared_error(y_test, predictions))
# print(model_history.history['loss'][0])

### Plot Results
plt.plot(predictions, label='Predicted')
plt.plot(y_test, label='Actual')
plt.legend()
plt.show()
