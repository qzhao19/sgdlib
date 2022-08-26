import numpy as np

class MSE(object):
    def __init__(self):
        pass

    def evaluate(self, X, y, W):
        num_samples = X.shape[0]
        y_hat = np.dot(X, W)
        return 0.5 * np.sum(np.power(y - y_hat, 2)) / num_samples

    def gradient(self, X, y, W):
        num_samples = X.shape[0]
        y_hat = np.dot(X, W)
        diff = y - y_hat
        return -1 / num_samples * np.dot(X.T, diff)
