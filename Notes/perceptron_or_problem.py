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

# for id, x in enumerate(sample_data):
#     input_vector = np.array(x)
#     activation_level = np.dot(input_vector, weights) + (bias_weight * 1)
#     print(f"Activation level for sample {id}: {activation_level}")
#     if activation_level > activation_threshold:
#         perceptron_output = 1
#     else:
#         perceptron_output = 0
#     print(f"Perceptron output for sample {id}: {perceptron_output}")

#     print("Predicted {}".format(perceptron_output))
#     print("Expected: {}".format(expected_results[id]))

### Adding logic to loop multiple time till we get all the answers right
for iteration_num in range(5):
    correct_answer = 0
    for id, x in enumerate(sample_data):
        input_vector = np.array(x)
        weights = np.array(weights)
        activation_level = np.dot(input_vector, weights) + (bias_weight * 1)
        if activation_level > activation_threshold:
            perceptron_output = 1
        else:
            perceptron_output = 0

        if perceptron_output == expected_results[id]:
            correct_answer += 1
        else:
            print(
                f"Wrong answer for sample {id} in iteration {iteration_num}. Updating weights..."
            )
            expected_output = expected_results[id]
            new_weights = []
            for i, x in enumerate(input_vector):
                new_weights.append(
                    weights[i] + (expected_results[id] - perceptron_output) * x
                )
            bias_weight += (expected_output - perceptron_output) * 1
            weights = np.array(new_weights)
    print(
        f"{correct_answer} correct answers out of {len(sample_data)} in iteration {iteration_num}"
    )
