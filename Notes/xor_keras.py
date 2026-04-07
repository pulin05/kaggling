import numpy as np
from keras import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

x_train = np.array[[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = np.array[[0], [1], [1], [0]]

model = Sequential()
num_neurons
