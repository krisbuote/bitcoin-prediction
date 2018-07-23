# A simple LSTM built from 'scratch' using the immensely useful Keras.

1. Have your bitcoin (or other time series data) in a .csv format.
2. Run pickle_it.py to turn the .csv into .pkl. use_cols argument to decide what data to include (e.g High, Low, Open, Close, etc)
3. Use run.py to run preprocess the data and train the model. Parameters can be played with in the run.py script.

lstm.py contains the neural network's architecture and functions to preprocess the data by scaling it, splitting it, and shifting it by n timesteps. If you choose 3 time steps = 3, each sample will be a np.array of size 3.

The experiment_xxx.py files contain code to test various values of parameters. For example, experiment_neurons.py compares [1, 4, 16, 64, 128] neurons in the LSTM across x epochs and y runs.

A lot more tuning and adjustments will be required to make this profitable, but this is a good starting point :)
