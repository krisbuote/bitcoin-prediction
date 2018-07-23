from lstm import model, scale, timeseries_to_supervised, split_and_reshape, data_raw
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

'''LATEST EXPERIMENT SHOWS T = 8 IS BEST'''

### Parameters not being experimented on
epochs = 50
runs = 3
batch_size = 16
# timesteps = 14 # how many timesteps the model looks in time
neurons = 1
data_dim = data_raw.shape[1]  # n_cols in data: only Bitcoin price currently
split_pct = 0.95  # percent of data in training
loading = False  # Set to True if loading a saved model
saving = True  # Set to True if you wish to save new model

experiment_timesteps = [1, 4, 8, 16, 32]

train_losses_in_experiments = np.zeros([len(experiment_timesteps),epochs])
test_losses_in_experiments = np.zeros(len(experiment_timesteps))
experiment = 0
seed = 0

for timesteps in experiment_timesteps:

    ### Preprocess Data
    scaler, raw_data_scaled = scale(data_raw, split_pct)  # Scale all of the data
    data_scaled = timeseries_to_supervised(raw_data_scaled, timesteps)  # turn it into timeshifted data
    x_train, y_train, x_test, y_test = split_and_reshape(data_scaled, raw_data_scaled, split_pct, timesteps, data_dim)  # Uses shifted data for x and original data for y

    this_model = model(neurons, timesteps, data_dim)
    train_loss_list = np.zeros([runs, epochs])
    test_loss_list = np.zeros(runs)

    for r in range(runs):
        np.random.seed(seed)

        model_history = this_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, shuffle=False)
        train_loss = model_history.history['loss']
        train_loss_list[r,:] = train_loss

        evaluate = this_model.evaluate(x_test, y_test, batch_size=batch_size)
        test_loss = evaluate[0]
        test_loss_list[r] = test_loss

        seed += 1

    average_train_loss = np.average(train_loss_list,0)
    average_test_loss = np.average(test_loss_list)

    train_losses_in_experiments[experiment, :] = average_train_loss
    test_losses_in_experiments[experiment] = average_test_loss
    print('Experiment %d Complete.' % (experiment))

    experiment += 1

np.save('./experiment_results/timestep_experiment_training_losses', train_losses_in_experiments)
np.save('./experiment_results/timestep_experiment_test_losses', test_losses_in_experiments)
print('All Experiments Finished and Data is Saved')


### Plot Results
# train_history.describe()

plt.plot(train_losses_in_experiments[0,:], label='Parameter_0')
plt.plot(train_losses_in_experiments[1,:], label='Parameter_1')
plt.plot(train_losses_in_experiments[2,:], label='Parameter_2')
plt.plot(train_losses_in_experiments[3,:], label='Parameter_3')
plt.plot(train_losses_in_experiments[4,:], label='Parameter_4')


# plt.plot(test_loss, label='Test Loss')
plt.legend()
plt.show()
plt.savefig('./experiment_results/timestep-training-losses.png')
# plt.plot(predictions, label='Predicted')
# plt.plot(y_test, label='Actual')
# plt.legend()
# plt.show()
