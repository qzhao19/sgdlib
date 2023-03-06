import numpy as np
from scipy import linalg

class HuberLoss(object):
    def __init__(self, delta):
        self.delta = delta
    
    def evaluate(self, X, y, W):
        if linalg.norm(y - np.dot(X, W)) <= self.delta:
            return 0.5 * linalg.norm(y - np.dot(X, W))
        else:
            return self.delta * linalg.norm((y - np.dot(X, W) - 0.5 * self.delta) , ord = 1)
    
    def gradient(self, X, y, W):
        if (linalg.norm(y - np.dot(X, W))) <= self.delta:
            grad = np.dot(X.T, (np.dot(X, W) - y)) / y.shape[0]
        else:
            grad =  self.delta * np.dot(X.T, np.sign(np.dot(X, W) - y)) / y.shape[0]

