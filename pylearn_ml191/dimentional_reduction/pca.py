import numpy as np 

class PCA(object):
    """ Principal Component Analysis """
    def __init__(self, n_components):
        super(PCA, self).__init__()
        self.n_components = n_components
            
    def fit(self, X):
        # Calculate mean 
        mean = np.mean(X, axis=0)
        
        # Subtract X by mean
        X_hat = X - mean
        
        # Calculate covariance matrix: S
        N = X.shape[0]
        S = (X_hat.T @ X_hat) / N 
        
        # Find eigenvectors and eigenvalues
        eigen_vals, eigen_vecs = np.linalg.eig(S)

        # Sort eigenvalues DESC
        sorted_idx = np.argsort(eigen_vals)[::-1]
        top_n_eigen_idx = sorted_idx[:self.n_components]
        
        # Save some result
        self.explained_variance_ = eigen_vals[top_n_eigen_idx]
        self.U = eigen_vecs.T[top_n_eigen_idx]
        self.mean = mean
        
    def transform(self, X):
        X_hat = X - self.mean
        return X_hat @ self.U.T
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, Z):
        return Z @ self.U + self.mean