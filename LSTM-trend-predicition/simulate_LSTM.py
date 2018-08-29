import pandas as pd
import numpy as np
from keras import Sequential, optimizers
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, LSTM, Bidirectional
from keras.models import load_model, save_model
import matplotlib.pyplot as plt

''' Load and put data into time series format
'''
raw_data = pd.read_csv('./data/preprocessed_6hr_data_lastYear_standardized_BB.csv')
data = raw_data.values # convert to numpy arrays
Y_raw = raw_data['Close'].values # The Close values will be used as labels

print(data.shape)

# Data size. Building the training data tensor X_construct will be shape (m, n_f, t)
timesteps = 1 # Number of 6 hour time chunks to treat as input feature
m = data.shape[0] - timesteps
n_f = data.shape[1]

''' The data is currently in dimension (n_examples, n_features).
We want to add a third dimension, for timesteps. Each new timestep entry will be the same data, shifted by
a timestep. For example, if there are 3 timesteps, the the third dimension will be size 3 where the first matrix
is the original data, the second matrix is the original data shifted up by one row, and the third matrix is the original
data shifted up by two rows. This would repeat t timesteps. '''
# The data will end in final shape of (m, n_f, t)
# Instantiate an array for constructing the 3 dimensional tensor.

X_construct = data[:-timesteps] # t timesteps must be shaved off the bottom of the data
X_construct = X_construct.reshape(m, n_f, 1) # reshape with a 3rd dimension for t
for t in range(1, timesteps):
    X_ti = np.copy(data[t:m+t]).reshape(m, n_f, 1) # original data shifted by t rows
    X_construct = np.concatenate((X_construct, X_ti), axis=2) # add this timesteps matrix to the tensor

print(X_construct.shape)

''' For Y, the labels, we want 1s and 0s to represent trend up vs trend down.
Y_change calculates the change in price, which is converted to 1s and 0s using numpy'''
Y_diff = raw_data['Close'].diff().iloc[timesteps:].values # Get the difference in price
Y = np.asarray((Y_diff > 0)).astype(int) # Convert to 1 (trend up), or 0 (trend down)
print(Y.shape)

X = X_construct.swapaxes(1,2) # shape (m, n_f, t) -> (m, t, n_f) for keras LSTM
Y = Y.reshape(len(Y), 1) # Column vector shape
print(X.shape)
print(Y.shape)

# Split training and testing data
test_split = int(0.9*len(Y))
X_train, X_test = X[:test_split], X[test_split:]
Y_train, Y_test = Y[:test_split], Y[test_split:]

ensemble = False
if ensemble:
    model1 = load_model('./models/model1.h5')
    model2 = load_model('./models/model2.h5')
    model3 = load_model('./models/model3.h5')

    # Take average of 3 model's predictions
    predictions = (model1.predict(X_test) + model2.predict(X_test) + model3.predict(X_test)) / 3

else:
    model = load_model('./models/LSTM-6hr-standardized.h5')
    print(model.summary())
    predictions = model.predict(X_test)

# Convert to 0s and 1s
predictions = np.array([round(i) for i in np.squeeze(predictions)]).astype(int)
Y_test = np.squeeze(Y_test)
print(predictions.shape)
print(Y_test.shape)

# Check accuracy
correct_counter = int(np.sum(predictions == Y_test))

print('{0} / {1} correct trend predictions ({2} %)'.format(correct_counter, len(predictions), round(correct_counter/len(predictions)*100, 2)))


## PLOT RESULTS ##
# Get the indicies for where the model predicted trend up (buy) vs trend down (sell)
index = np.arange(len(predictions))
buy_indicies = np.where(predictions > 0.5)[0]
sell_indicies = np.where(predictions < 0.5)[0]


trend = np.squeeze(raw_data['Close'][-(len(predictions)+timesteps):-timesteps].values) # The close values using to make predictions
buys = trend[buy_indicies]
sells = trend[sell_indicies]
plt.plot(index, trend, label='Actual')
plt.plot(buy_indicies, buys, 'go', label='Buys')
plt.plot(sell_indicies, sells, 'ro', label='Sells')
plt.legend()
plt.title('{0} / {1} correct trend predictions ({2} %)'.format(correct_counter, len(predictions), round(correct_counter/len(predictions)*100, 2)))
plt.show()



