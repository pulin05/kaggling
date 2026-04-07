import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

np.random.seed(42)
tf.random.set_seed(42)

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

model = Sequential()
num_neurons = 10
model.add(Dense(num_neurons, input_dim=2))
model.add(Activation("tanh"))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.summary()

sgd = SGD(learning_rate=0.1)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5000, verbose=0)
print(model.predict(x_train))


## save model and weights
import h5py

model_structure = model.to_json()
with open("xor_model.json", "w") as f:
    f.write(model_structure)

model.save_weights("xor_model.h5")
