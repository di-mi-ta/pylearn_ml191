import numpy as np
from math import sqrt

# Mean-squared errors
def mse(targets, predictions):
    N = targets.shape[0]
    res = 0
    for y_truth, y_pred in zip(targets, predictions):
        res += (y_truth - y_pred) ** 2
    return 1.0/N * res