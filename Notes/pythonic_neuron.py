import numpy as np

example_input = [1, 0.2, 0.1, 0.05, 0.2]
example_weights = [0.2, 0.12, 0.4, 0.6, 0.90]

input_vector = np.array(example_input)
weight_vector = np.array(example_weights)
bias_weight = 0.2

activation_level = np.dot(input_vector, weight_vector) + (bias_weight * 1)
print(activation_level)

threshold = 0.5
if activation_level > threshold:
    perceptron_output = 1
else:
    perceptron_output = 0
print(f"Final output: {perceptron_output}")

###########

expected_output = 0
new_weights = []
for i, x in enumerate(example_input):
    new_weights.append(weight_vector[i] + (expected_output - perceptron_output) * x)
    print(f"Updated weight {i}: {new_weights[i]}")
