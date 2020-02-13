import numpy as np 
from ..functional import softmax

# Helper function 
def to_categorical(Y_one_hot):
    y = np.argmax(Y_one_hot, axis=1)
    return y

def one_hot_encoding(y, num_classes):
    num_samples = y.shape[0]
    Y = np.zeros((num_samples, num_classes))
    for idx, val in enumerate(y):
        Y[idx][int(val)] = 1
    return Y 

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

def batch_data(X_train, y_train, batch_size=64):
    y_train = np.reshape(y_train, (y_train.shape[0],1))
    data_train = np.concatenate([X_train, y_train], axis=1)
    data_train_shuffled = np.random.permutation(data_train)
    num_batchs = data_train.shape[0] // batch_size
    batchs = [data_train_shuffled[i: (i+1)*batch_size, :] for i in range(num_batchs)]
    data_loader = iter(batchs)
    return data_loader


class SoftmaxRegression(object):
    """ Softmax Regression """
    def __init__(self, num_classes, use_features_extractor=True):
        super(SoftmaxRegression, self).__init__()
        self.use_features_extractor = use_features_extractor
        self.num_classes = num_classes 
        
    def fit(self, X_train, y_train,
            X_val, y_val, batch_size=1, lr=0.01, verbose=10):
        
        if self.use_features_extractor:
            X_train = feature_extractor(X_train)
            X_val = feature_extractor(X_val)
            
        # Initialize weights: 
        dim_features = X_train.shape[1]
        W = np.random.randn(self.num_classes, dim_features)
        
        step = 0 
        max_steps = 10000
        min_W_diff = 1e-4
        losses_train = []
        losses_val = []
        best_W = None 
        best_val_loss = float('inf')
        EPSILON = 1e-4
        steps = []
        
        Y_val = one_hot_encoding(y_val, self.num_classes)
        data_loader = batch_data(X_train, y_train, batch_size)
        
        while True:
            step += 1
            
            # Learning rate decay
            if step % 100 == 0:
                # Divide learning rate by 2 after each 100 iterations
                lr *= 0.5
                
            try:
                batch = next(data_loader)
                X = batch[:, :-1]
                y = batch[:, -1]
                Y = one_hot_encoding(y, self.num_classes)
            except:
                data_loader = batch_data(X_train, y_train, batch_size)
                batch = next(data_loader)
                X = batch[:, :-1]
                y = batch[:, -1]
                Y = one_hot_encoding(y, self.num_classes)
                
            # Forward phase
            Y_hat = softmax(X @ W.T)

            # Calculate CE loss
            loss = (-1.0/batch_size) * np.sum(Y * np.log(Y_hat + EPSILON))

            # Backward phase
            # Calculate gradients on W 
            dW =(Y_hat - Y).T @ X
             
            # Update weights 
            W_new = W - lr * dW 
            W_diff = np.linalg.norm(W_new - W)
            
            if step % verbose == 0:
                Y_preds = softmax(X_val @ W.T)
                val_loss = (-1.0/Y_preds.shape[0]) * np.sum(Y_val * np.log(Y_preds + EPSILON))
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

            
                
    def eval(self, X, categorical=False):
        if self.use_features_extractor:
            X = feature_extractor(X)
        if categorical:
            return to_categorical(softmax(X @ self.W.T))
        else:
            return softmax(X @ self.W.T)