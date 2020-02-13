import numpy as np
from math import sqrt
from ..metrics import rmse

# Helper functions
def features_extractor(X_raw, M):
    X_features = np.ones(X_raw.shape)
    for m in range(1, M + 1):
        X_features = np.concatenate([X_features, X_raw ** m], axis=1)
    return X_features

# Main class
class LinearRegression():
    def __init__(self):
        self.w = None
        
    def fit(self, X, y, M, weight_decay):
        X = features_extractor(X, M)
        self.M = M
        self.weight_decay = weight_decay
        self.w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + self.weight_decay * np.eye(self.M + 1)), X.T), y)
    
    def eval(self, X):
        return [np.dot(self.w.T, x).item() for x in features_extractor(X, self.M)]