## OR problem
sample_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]  # False, False # False, True # True, False # True, True
expected_results = [
    0,
    1,
    1,
    1,
]  # (False OR False) gives False # (False OR True ) gives True # (True OR False) gives True
# (True OR True ) gives True

activation_threshold = 0.5

from random import random
import numpy as np

weights = np.random.random(2) / 1000
print(weights)

bias_weight = np.random.random() / 1000
print(bias_weight)

activation_threshold = 0.5

for id, x in enumerate(sample_data):
    input_vector = np.array(x)
    activation_level = np.dot(input_vector, weights) + (bias_weight * 1)
    print(f"Activation level for sample {id}: {activation_level}")
    if activation_level > activation_threshold:
        perceptron_output = 1
    else:
        perceptron_output = 0
    print(f"Perceptron output for sample {id}: {perceptron_output}")

    print("Predicted {}".format(perceptron_output))
    print("Expected: {}".format(expected_results[id]))
