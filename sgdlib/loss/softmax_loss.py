import numpy as np
from ..common import _softmax

class Softmax(object):
    def __init__(self, mu = 0.0):
        self._mu = mu

    def evaluate(self, X, y, W):
        Z = - X @ W
        num_samples = X.shape[0]
        loss = 1 / num_samples * (np.trace(X @ W @ y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        return loss

    def gradient(self, X, y, W):
        Z = - X @ W
        P = _softmax(Z, axis=1)
        num_samples = X.shape[0]
        gd = 1 / num_samples * (X.T @ (y - P)) + 2 * self._mu * W
        return gd
