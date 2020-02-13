import numpy as np

def sigmoid(s):
    return 1/(1 + np.exp(-s))