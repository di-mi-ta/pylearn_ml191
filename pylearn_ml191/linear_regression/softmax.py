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

def batch_data(X, y, batch_size=64, shuffle=True):
    y = np.reshape(y, (y.shape[0],1))
    data = np.concatenate([X, y], axis=1)
    if shuffle:
        data = np.random.permutation(data)
    num_batchs = data.shape[0] // batch_size
    batchs = [data[i*batch_size: (i+1)*batch_size, :] for i in range(num_batchs)]
    return batchs


class SoftmaxRegression(object):
    """ Softmax Regression """
    def __init__(self, num_classes, use_features_extractor=True):
        super(SoftmaxRegression, self).__init__()
        self.use_features_extractor = use_features_extractor
        self.num_classes = num_classes 
        
    def fit(self, X_train, y_train,
            X_val, y_val, 
            batch_size=1, 
            lr=0.01, 
            verbose=10, 
            max_steps=10000,
            step_to_lr_decay=500, 
            lr_decay=0.1,
            min_W_diff=1e-4,
            eps=1e-4):
        
        if self.use_features_extractor:
            X_train = feature_extractor(X_train)
            X_val = feature_extractor(X_val)
            
        # Initialize weights: 
        dim_features = X_train.shape[1]
        W = np.random.randn(self.num_classes, dim_features)
        
        step = 0 
        losses_train = []
        losses_val = []
        best_W = None 
        best_val_loss = float('inf')
        EPSILON = eps
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
                Y = one_hot_encoding(y, self.num_classes)
            except:
                data_loader = iter(batch_data(X_train, y_train, batch_size))
                batch = next(data_loader)
                X = batch[:, :-1]
                y = batch[:, -1]
                Y = one_hot_encoding(y, self.num_classes)

            # Forward phase
            Y_hat = softmax(X @ W.T)

            # Calculate CE loss
            loss = -(1.0 / batch_size) * np.sum(Y * np.log(Y_hat + EPSILON))

            # Backward phase
            # Calculate gradients on W 
            dW =(Y_hat - Y).T @ X
             
            # Update weights 
            W_new = W - lr * dW 
            W_diff = np.linalg.norm(W_new - W)
            
            if step % verbose == 0:
                val_loss = 0
                for batch_idx, batch in enumerate(val_loader):
                    batch_features = batch[:, :-1]
                    batch_label = batch[:, -1]
                    batch_label = one_hot_encoding(batch_label, num_classes=self.num_classes)
                    #
                    y_preds = softmax(batch_features @ W.T)
                    val_loss += -(1.0 / batch_size) * np.sum(batch_label * np.log(y_preds + EPSILON))
                val_loss /= num_batch_val

                # Log loss to track training history
                losses_train.append(loss)
                losses_val.append(val_loss)
                steps.append(step)

                # Save best model
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    best_W = W
                print("[Step: {}] Train-loss: {}, Val-loss: {}".format(step, loss, val_loss))

            if step == max_steps or W_diff < min_W_diff:
                # Stop training 
                print("Stop training at step: {}, best val loss: {}".format(step, best_val_loss))
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