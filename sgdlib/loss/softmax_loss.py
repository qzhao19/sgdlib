import numpy as np
from ..common import _softmax

class Softmax(object):
    def __init__(self, mu = 0.0):
        self._mu = mu

    def evaluate(self, X, y, w):
        loss = 0 
        num_samples, _ = X.shape
        xw = X.dot(w)
        xw -= np.max(xw)

        exp_xw = np.exp(xw)
        sum_exp_xw = np.sum(exp_xw, axis=1)
        
        argmax_exp_xw = exp_xw[range(num_samples), y]
        loss = -np.sum(np.log(argmax_exp_xw / sum_exp_xw))
        loss /= num_samples
        loss += 0.5 * self._mu * np.sum(w * w)
        
        return loss

    def gradient(self, X, y, w):
        grad = np.zeros_like(w)
        num_samples, _ = X.shape
        xw = X.dot(w)
        xw -= np.max(xw)

        exp_xw = np.exp(xw)
        sum_exp_xw = np.sum(exp_xw, axis=1)
        normalized_exp_xw = exp_xw / sum_exp_xw[:, np.newaxis]

        normalized_exp_xw[range(num_samples), y] -= 1
        grad = X.T.dot(normalized_exp_xw)
        grad /= num_samples
        grad += self._mu * w

        return grad

