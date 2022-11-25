import numpy as np

class Rosenbrock(object):
    def __init__(self, n = 100):
        self.n = n
    
    def __compute(self, x):
        self.fx = 0.0
        self.grad = np.zeros((self.n))
        for i in range(0, self.n, 2):
            t1 = 1.0 - x[i]
            t2 = 10.0 * (x[i + 1] - x[i] * x[i])
            self.grad[i + 1] = 20.0 * t2
            self.grad[i] = -2.0 * (x[i] * self.grad[i + 1] + t1)
            self.fx += t1 * t1 + t2 * t2
    
    def evaluate(self, X, y, W):
        self.fx, self.grad = self.__compute(W)
        return self.fx

    def gradient(self, X, y, W):
        return self.grad