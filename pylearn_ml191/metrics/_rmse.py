from ._mse import mse 
from math import sqrt

# Root mean-squared errors
def rmse(targets, predictions):
    return sqrt(mse(targets, predictions))