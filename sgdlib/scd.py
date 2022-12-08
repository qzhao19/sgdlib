import numpy as np
from .base import BaseOptimizer

class SCD(BaseOptimizer):
    def __init__(self, x0, 
        loss_func, 
        max_iters = 50,
        rho = 0.15, 
        alpha = 0.001, 
        shuffle = True, 
        verbose = True):
        super(SCD, self).__init__(x0 = x0, 
                loss_func = loss_func, 
                max_iters = max_iters, 
                shuffle = shuffle,
                verbose = verbose)
        self.rho = rho
        self.alpha = alpha

    def optimize(self, X, y):
        num_features = X.shape[1]
        X_y = np.c_[X, y]
        i = 0
        eta = 0
        for iters in range(self.max_iters):
            if self.shuffle:
                np.random.shuffle(X_y)
                X = X_y[:, :-1]
                y = X_y[:, -1:].squeeze()
            grad = self.loss_func.gradient(X, y, self.x0)
            pred_descent = 0
            best_descent = -1
            best_index = 0
            best_eta = 0
            for i in range(num_features):
                if self.x0[i] - grad[i] /self.rho > self.alpha / self.rho:
                    eta = -grad[i] /self.rho - self.alpha / self.rho
                elif self.x0[i] - grad[i] /self.rho < -self.alpha / self.rho:
                    eta = -grad[i] /self.rho + self.alpha / self.rho
                else:
                    eta = -self.x0[i]

                pred_descent = -eta * grad[i] - self.rho / 2 * eta * eta - self.alpha * abs(self.x0[i] + eta) + self.alpha * abs(self.x0[i])

                if pred_descent > best_descent:
                    best_descent = pred_descent
                    best_index = i
                    best_eta = eta

            i = best_index
            eta = best_eta
            self.x0[i] += eta

            if self.verbose:
                if iters % 5 == 0:
                    w_norm = np.linalg.norm(self.x0, ord = 2)
                    loss = self.loss_func.evaluate(X, y, self.x0)
                    print("-- Epoch = {}, weight norm = {}, loss value = {}".format(iters, w_norm, (loss/num_features)))
        
        return self.x0