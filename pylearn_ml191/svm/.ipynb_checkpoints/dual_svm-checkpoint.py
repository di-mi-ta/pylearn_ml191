import numpy as np 
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel, sigmoid_kernel

# Kernel SVM 
def kernel_func(X1, X2, kernel_name, gamma, d, r):
    if kernel_name == 'rbf':
        return rbf_kernel(X1, X2, gamma = gamma)
    elif kernel_name == 'polynomial':
        return polynomial_kernel(X1, X2, gamma = gamma, degree = d, coef0 = r)
    elif kernel_name == 'sigmoid':
        return sigmoid_kernel(X1, X2, gamma = gamma, coef0 = r)
    elif kernel_name == 'linear':
        return linear_kernel(X1, X2)
    else:
        raise NotImplementedError

class KernelSVM(object):
    """ Kernel SVM Classifier"""
    def __init__(self, kernel='linear', C=0, atol=1e-4, gamma=0.2, r=0, d=0):
        super(KernelSVM, self).__init__()
        self.C = C
        self.gamma = gamma # for polynomial kernel + rbf kernel + sigmoid kernel 
        self.r = r # for polynomial kernel + sigmoid kernel 
        self.d = d # for polynomial kernel
        self.kernel = kernel
        self.atol = atol
            
    def fit(self, X, t):
        NUM_SAMPLES = X.shape[0]
        DIM_FEATURES = X.shape[1]
        # Define matrixs
        K_gram = kernel_func(X, X, self.kernel, self.gamma, self.d, self.r)
        Y = t @ t.T 
        K = K_gram * Y 
        Q = matrix(K) 
        p = matrix(-np.ones((NUM_SAMPLES, 1)))  
        G = matrix(np.concatenate((-np.eye(NUM_SAMPLES), np.eye(NUM_SAMPLES)), axis=0))  
        h = matrix(np.concatenate((np.zeros(NUM_SAMPLES), np.full(NUM_SAMPLES, self.C)), axis=0))  
        A = matrix(t.reshape(1, -1))  
        b = matrix(np.zeros(1))  
        
        # Solve
        solvers.options["show_progress"] = False 
        sol = solvers.qp(Q, p, G, h, A, b)
        alphas = np.array(sol['x'])
        
        # Get support vectors
        S = np.where(alphas > self.atol)[0]
        alphas_1 = alphas[S]
        XS = X[S, :]
        tS = t[S, :]
        aS = alphas[S] * tS
        
        M = np.where(abs(alphas_1 - self.C) > self.atol)[0]
        XM = XS[M, :]
        tM = tS[M, :]
        aM = alphas_1[M] * tM
        
        # Get and save b for predict
        Kms = kernel_func(XM, XS, self.kernel, self.gamma, self.d, self.r)
        w = np.sum(aS * XS, axis = 0).reshape(1, -1) 
        b = np.mean(tM.T - Kms @ aS).reshape(1, )
        self.w = w
        self.b = b
        
        # Save support vectors 
        self.support_vectors_ = XS
        self.target_support_vectors_ = tS
        self.aS = aS
        self.margin_vectors_ = XM
        self.target_margin_vectors_ = tM
        self.aM = aM 
    
    def predict(self, Xb):
        aS = self.aS
        Xs = self.support_vectors_
        tS = self.target_support_vectors_
        
        aM = self.aM
        Xm = self.margin_vectors_
        tM = self.target_margin_vectors_
        
        Kms = kernel_func(Xm, Xs, self.kernel, self.gamma, self.d, self.r)
        Kbs = kernel_func(Xb, Xs, self.kernel, self.gamma, self.d, self.r)
            
        y = Kbs @ aS + self.b
        labels_preds = np.sign(y)
        return labels_preds
    
    def functional_distance(self, Xb):
        aS = self.aS
        Xs = self.support_vectors_
        tS = self.target_support_vectors_
        
        aM = self.aM
        Xm = self.margin_vectors_
        tM = self.target_margin_vectors_
        
        Kms = kernel_func(Xm, Xs, self.kernel, self.gamma, self.d, self.r)
        Kbs = kernel_func(Xb, Xs, self.kernel, self.gamma, self.d, self.r)
            
        y = Kbs @ aS + self.b    
        return y