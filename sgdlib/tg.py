import numpy as np
from .base import BaseOptimizer

class TruncatedGradient(BaseOptimizer):
    def __init__(self, 
        x0,
        loss_func, 
        lr_decay, 
        regularizer, 
        tol = 0.001, 
        alpha = 0.0001,
        l1_ratio = 0.15,
        batch_size = 16, 
        max_iters = 2000, 
        num_iters_no_change = 5, 
        shuffle = True, 
        verbose = True):
            super(TruncatedGradient, self).__init__(x0 = x0, 
                loss_func = loss_func, 
                lr_decay = lr_decay, 
                regularizer = regularizer, 
                tol = tol, 
                batch_size = batch_size, 
                max_iters = max_iters,
                num_iters_no_change = num_iters_no_change,
                shuffle = shuffle,
                verbose = verbose)
            self.alpha = alpha
            self.l1_ratio = l1_ratio
            
    def update(self, weight, cum_l1, max_cum_l1):
        w = weight
        num_features = len(weight)
        # cum_l1 = np.zeros((num_features,))
        
        for j in range(num_features):
            w_j = w[j]
            if w_j > 0.0:
                w[j] = max(0.0, w_j - (max_cum_l1 + cum_l1[j]))
            elif w_j < 0.0:
                w[j] = min(0.0, w_j + (max_cum_l1 - cum_l1[j]))
            else:
                w[j] = w_j

            cum_l1[j] += w[j] - w_j

        return w

    def optimize(self, X, y):
        num_features = X.shape[1]
        X_y = np.c_[X, y]

        # total_loss = 0.0
        best_loss = np.Inf

        cum_l1 = np.zeros((num_features,))
        max_cum_l1 = 0.0
        
        is_converged = False
        no_improvement_count = 0

        for iters in range(self.max_iters):
            np.random.shuffle(X_y)
            error_history = []
            lr = self.lr_decay.compute(iters)
            # print(X_y.shape[0] // self.batch_size)
            for index in range(X_y.shape[0] // self.batch_size):
                X_y_batch = X_y[self.batch_size * index : self.batch_size * (index + 1)]
                X_batch = X_y_batch[:, :-1]
                y_batch = X_y_batch[:, -1:]

                grad = self.loss_func.gradient(X_batch, y_batch, self.x0)

                np.clip(grad, self.MIN_DLOSS, self.MAX_DLOSS, out = grad)

                self.x0 = self.x0 - lr * grad
                
                max_cum_l1 += self.l1_ratio * lr * self.alpha
                self.x0 = self.update(self.x0, cum_l1, max_cum_l1)

                loss = self.loss_func.evaluate(X_batch, y_batch, self.x0)
                error_history.append(loss)
            
            sum_loss = np.sum(error_history)
            # avg_loss = np.mean(error_history)

            if self.tol > -np.Inf and sum_loss > best_loss - self.tol * self.batch_size:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            if sum_loss < best_loss:
                best_loss = sum_loss

            if no_improvement_count >= self.num_iters_no_change:
                is_converged = True
                self.opt_x = self.x0
                break
                
            if iters % 5 == 0:
                print("--Epoch = {}, average loss value = {}".format(str(iters), str(sum_loss/self.batch_size)))

        if not is_converged:
            raise ValueError("Not converged!")

        return self.opt_x
        