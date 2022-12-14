import numpy as np
from scipy.special import softmax

class LogLoss(object):
    def __init__(self, mu):
        self._mu = mu

    def evaluate(self, X, y, W):
        Z = - X @ W
        num_samples = X.shape[0]
        loss = 1 / num_samples * (np.trace(X @ W @ y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        return loss

    def gradient(self, X, y, W):
        Z = - X @ W
        P = softmax(Z, axis=1)
        num_samples = X.shape[0]
        gd = 1 / num_samples * (X.T @ (y - P)) + 2 * self._mu * W
        return gd
