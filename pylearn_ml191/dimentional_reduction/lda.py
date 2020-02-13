import numpy as np 
import scipy

class LDA(object):
    """ Linear Discriminant Analysis (LDA) """
    
    def __init__(self, n_components):
        super(LDA, self).__init__()
        self.n_components = n_components
            
            
    def fit(self, X, y):
        """
        Fit LDA model according to the given training data and parameters.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array, shape (n_samples,)
            Target values.

        """
        
        self.classes = np.unique(y)
        self.num_classes = self.classes.shape[0]
        self.n_samples, feat_dim = X.shape

        if self.n_samples < self.num_classes:
            raise ValueError("The number of samples must be more than the number of classes.")
        
        # Building SB, SW
        self.m = np.mean(X, axis=0, keepdims=True)
        SB = np.zeros((feat_dim, feat_dim))
        SW = np.zeros((feat_dim, feat_dim))
        
        for c in self.classes:
            Xc = X[y == c, :]
            Nc = Xc.shape[0]
            mc = np.mean(Xc, axis=0, keepdims=True)
            SB += Nc * ((mc - self.m).T @ (mc - self.m))
            SW += (Xc - mc).T @ (Xc - mc)
            
        # Calc A = SW^-1 @ SB
        A = np.linalg.pinv(SW) @ SB 
        
        # Find eigenvectors and eigenvalues
        eigen_vals, eigen_vecs = scipy.linalg.eig(SB, SW)

        # Sort eigenvalues DESC
        sorted_idx = np.argsort(eigen_vals)[::-1]
        top_n_eigen_idx = sorted_idx[:self.n_components]
        
        # Save some result
        self.W = eigen_vecs.T[top_n_eigen_idx]
        
    def transform(self, X):
        """
        Project data to maximize class separation.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Transformed data.
        
        """
        X_hat = X - self.m
        return X_hat @ self.W.T
    
    def fit_transform(self, X, y):
        """
        Fit LDA model according to the given training data and parameters.
        Then, project data to new space above.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array, shape (n_samples,)
            Target values.
        
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X, y)
        return self.transform(X)
    
    def inverse_transform(self, Z):
        """
        Reconstruct original data from transformed data.
        
        Parameters
        ----------
        Z: array-like, shape (n_samples, n_components)
        
        Returns:
        -------
        X_org: array, shape (n_samples, n_features)
        
        """
        return Z @ self.W + self.m