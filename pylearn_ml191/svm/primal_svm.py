import numpy as np 
from cvxopt import matrix, solvers

class PrimalSVM(object):
    """ Primal Problem in SVM """
    def __init__(self, M=3, EPSILON=1e-7):
        super(PrimalSVM, self).__init__()
        self.EPSILON = EPSILON
        self.M = 3

    def fit(self, X, t):
        """
            Args:
                X: input features (#num_samples, features_dim)
                t: target (#num_samples, 1)
        """
        N = X.shape[0]
        K = np.eye(self.M)
        K[-1, -1] = 0
        K = matrix(K)
        p = matrix(np.zeros((self.M, 1)))
        h = matrix(-np.ones((N, 1)))
        G = - np.concatenate((X, np.ones((N, 1))), axis = 1) * t
        G = matrix(G)
        # Solve 
        solvers.options["show_progress"] = False 
        sol = solvers.qp(K, p, G, h)
        wb = np.array(sol['x'])
        self.w = wb[:-1].reshape(1, -1)
        self.b = wb[-1]
        return self.w, self.b
    
    
    def get_support_vectors(self, X):
        # Get some points, which subject to abs(w.T @ x + b) = 1
        y = abs(X @ self.w.T + self.b ) - 1
        idx_spv = np.where(y < self.EPSILON)[0]
        return X[idx_spv, :]
    
    def predict(self, X):
        """
            Parameters:
                X: input_features (#num_samples, #features_dim) 
            Return: 
                y: predicted vector  (#num_samples, 1)
        """
        y = X @ self.w.T + self.b
        return np.sign(y)