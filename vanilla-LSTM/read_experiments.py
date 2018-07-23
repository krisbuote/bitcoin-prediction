import numpy as np
import matplotlib.pyplot as plt
#Rows are results for 1, 4, 16, 64, 128 neurons
training_loss_per_neurons = np.load('./experiment_results/neuron_experiment_training_losses.npy')
test_loss_per_neurons = np.load('./experiment_results/neuron_experiment_test_losses.npy')
one = training_loss_per_neurons[0,:]
four = training_loss_per_neurons[1,:]
sixteen = training_loss_per_neurons[2,:]
sixtyfour= training_loss_per_neurons[3,:]
onetwentyeight= training_loss_per_neurons[4,:]

test_loss = np.load('./experiment_results/neuron_experiment_test_losses.npy')
test_loss_time = np.load('./experiment_results/timestep_experiment_test_losses.npy')

print(test_loss)
print(test_loss_time)

plt.plot(one, label = 'one')
plt.plot(four, label = 'four')
plt.plot(sixteen, label = 'sixteen')
plt.plot(sixtyfour, label = 'sixtyfour')
plt.plot(onetwentyeight, label = 'onetwentyeight')
plt.legend()
plt.show()