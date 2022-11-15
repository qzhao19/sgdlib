import numpy as np

class SCD(object):
    def __init__(self, loss_func, max_iters = 50, rho = 1.0, alpha = 0.001):
        self.max_iters = 50
        self.loss_func = loss_func
        self.rho = 1.0
        self.alpha = 0.001

    def optimize(self, X, y):
        num_features = X.shape[1]
        X_y = np.c_[X, y]

        # W = np.random.rand(X.shape[1], 1)
        W = np.zeros((X.shape[1]))
        i = 0
        eta = 0
        for iters in range(self.max_iters):
            grad = self.loss_func.gradient(X, y, W)
            print(grad)

            pred_descent = 0
            best_descent = -1

            best_index = 0
            best_eta = 0
            for i in range(num_features):
                if W[i] - grad[i] /self.rho > self.alpha / self.rho:
                    eta = -grad[i] /self.rho - self.alpha / self.rho
                elif W[i] - grad[i] /self.rho < -self.alpha / self.rho:
                    eta = -grad[i] /self.rho + self.alpha / self.rho
                else:
                    eta = -W[i]

                pred_descent = -eta * grad[i] - self.rho / 2 * eta * eta - self.alpha * abs(W[i] + eta) + self.alpha * abs(W[i])

                if pred_descent > best_descent:
                    best_descent = pred_descent
                    best_index = i
                    best_eta = eta

            i = best_index
            eta = best_eta
            W[i] += eta
        
        return W