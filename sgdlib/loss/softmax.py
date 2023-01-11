import numpy as np

def _softmax(x, axis = None):
    if len(x.shape) != 2:
        raise ValueError("Input ndarray should be 2 dimension.")
    
    if axis is None:
        c = np.max(x, axis = axis)
    else:
        c = np.expand_dims(np.max(x, axis = axis), axis)
    exp_x = np.exp(x - c)

    if axis is None:
        div = np.sum(exp_x, axis = axis)
    else:
        div = np.expand_dims(np.sum(exp_x, axis = axis), axis)
    return exp_x / div


class Softmax(object):
    def __init__(self, mu):
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
