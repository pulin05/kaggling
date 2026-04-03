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

for 