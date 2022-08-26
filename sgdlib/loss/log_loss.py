import numpy as np

def _sigmoid(x):
    return np.where(x >= 0, 
        1 / (1 + np.exp(-x)), 
            np.exp(x) / (1 + np.exp(x)))

class LogLoss(object):
    def __init__(self):
        pass

    def evaluate(self, X, y, W):
        # np.sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)) / len(y_hat)
        y_hat = _sigmoid(np.dot(X, W))
        return np.sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)) / len(X)

    def gradient(self, X, y, W):
        grad = np.dot(X.T, _sigmoid(np.dot(X, W)) - y) / len(X)
        
        return grad
