
import numpy as np 
from ..functional import sigmoid

def feature_extractor_for_one_sample(x_raw):
    # M = 4
    x_0 = 1
    x_1 = x_raw[1] 
    x_2 = x_raw[2] 
    x_3 = x_raw[1] ** 2
    x_4 = x_raw[2] ** 2
    x_5 = x_raw[1] * x_raw[2]
    x_6 = x_raw[1] ** 3
    x_7 = x_raw[2] ** 3
    x_8 = x_raw[1]**2 * x_raw[2]
    x_9 = x_raw[1] * x_raw[2]**2
    x_10 = x_raw[1]**4
    x_11 = x_raw[1]**4
    x = np.array([x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11])
    return x

def feature_extractor(X_raw):
    return np.array([feature_extractor_for_one_sample(x) for x in X_raw])

def batch_data(X, y, batch_size=64, shuffle=True):
    y = np.reshape(y, (y.shape[0],1))
    data = np.concatenate([X, y], axis=1)
    if shuffle:
        data = np.random.permutation(data)
    num_batchs = data.shape[0] // batch_size
    batchs = [data[i*batch_size: (i+1)*batch_size, :] for i in range(num_batchs)]
    return batchs

class LogisticRegression(object):
    """ Simple Logistic Classifier"""
    def __init__(self, use_features_extractor=True):
        super(LogisticRegression, self).__init__()
        self.use_features_extractor = use_features_extractor
        
    def fit(self, X_train, y_train,
            X_val, y_val, 
            batch_size=1, 
            lr=0.01, 
            verbose=10, 
            max_steps=10000,
            step_to_lr_decay=500, 
            lr_decay=0.1):
        
        if self.use_features_extractor:
            X_train = feature_extractor(X_train)
            X_val = feature_extractor(X_val)
            
        # Initialize weights: 
        dim_features = X_train.shape[1]
        W = np.random.randn(dim_features, 1)
        
        step = 0 
        min_W_diff = 1e-4
        losses_train = []
        losses_val = []
        best_W = None 
        best_val_loss = float('inf')
        EPSILON = 1e-4
        steps = []
        
        train_loader = batch_data(X_train, y_train, batch_size, shuffle=True)
        val_loader = batch_data(X_val, y_val, batch_size, shuffle=False)
        
        num_batch_train = len(train_loader)
        num_batch_val = len(val_loader)
        
        while True:
            step += 1
            
            # Learning rate decay
            if step % step_to_lr_decay == 0:
                lr *= lr_decay
                
            try:
                batch = next(data_loader)
                X = batch[:, :-1]
                y = batch[:, -1]
                y = np.reshape(y, (batch_size, -1))
            except:
                data_loader = iter(batch_data(X_train, y_train, batch_size, shuffle=True))
                batch = next(data_loader)
                X = batch[:, :-1]
                y = batch[:, -1]
                y = np.reshape(y, (batch_size, -1))
                
            # Forward phase
            y_hat = sigmoid(X @ W)
            
            # Calculate CE loss 
            loss = -(1.0 / batch_size) * np.sum(y * np.log(y_hat + EPSILON) + (1-y) * np.log(1 - y_hat + EPSILON))

            # Backward phase
            # Calculate gradients on W 
            dW = X.T @ (y_hat - y)
    
            # Update weights 
            W_new = W - lr * dW 
            
            W_diff = np.linalg.norm(W_new - W)
            
            if step % verbose == 0:
                val_loss = 0
                for batch_idx, batch in enumerate(val_loader):
                    batch_features = batch[:, :-1]
                    batch_label = batch[:, -1]
                    batch_label = np.reshape(batch_label, (batch_size, -1))
                    #
                    y_preds = sigmoid(batch_features @ W)
                    val_loss += (-1.0 / batch_size) * np.sum(batch_label * np.log(y_preds + EPSILON) + (1 - batch_label) * np.log(1 - y_preds + EPSILON))
                val_loss /= num_batch_val
                #
                losses_train.append(loss)
                losses_val.append(val_loss)
                steps.append(step)
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    best_W = W
                print("[Step: {}] Train-loss: {:0.4f}, Val-loss: {:0.4f}".format(step, loss, val_loss))
            
            if step == max_steps or W_diff < min_W_diff:
                # Stop training 
                print("Stop training at step: {}, best val loss: {:0.4f}".format(step, best_val_loss))
                history = {
                    "steps" : steps,
                    "train_losses" : losses_train,
                    "val_losses" : losses_val
                }

                # Return model with best loss on val set 
                self.W = best_W

                return history
            
            W = W_new
            
    def predict(self, X):
        if self.use_features_extractor:
            X = feature_extractor(X)
        prob = sigmoid(X @ self.W)
        return np.array([1 if p > 0.5 else 0 for p in prob])