import numpy as np
from .base import BaseOptimizer

class SGD(BaseOptimizer):
    """Mini-batch stochastic gradient descent algorithm

    Parameters
    ----------
    loss : class object
        _description_
    """
    def __init__(self, 
        x0,
        loss_func, 
        lr_decay, 
        regularizer, 
        tol = 0.001, 
        batch_size = 16, 
        max_iters = 2000, 
        num_iters_no_change = 5, 
        shuffle = True, 
        verbose = True):
            super(SGD, self).__init__(x0 = x0, 
                loss_func = loss_func, 
                lr_decay = lr_decay, 
                regularizer = regularizer, 
                tol = tol, 
                batch_size = batch_size, 
                max_iters = max_iters,
                num_iters_no_change = num_iters_no_change,
                shuffle = shuffle,
                verbose = verbose)

    def optimize(self, X, y):
        best_loss = np.Inf
        # self.x0 = np.random.rand(X.shape[1], 1)
        
        X_y = np.c_[X, y]
        is_converged = False
        no_improvement_count = 0

        for iters in range(self.max_iters):
            if self.shuffle:
                np.random.shuffle(X_y)
            error_history = []
            lr = self.lr_decay.compute(iters)
            for index in range(X_y.shape[0] // self.batch_size):
                X_y_batch = X_y[self.batch_size * index : self.batch_size * (index + 1)]
                X_batch = X_y_batch[:, :-1]
                y_batch = X_y_batch[:, -1:]

                grad = self.loss_func.gradient(X_batch, y_batch, self.x0)
                grad += self.regularizer.gradient(self.x0, self.batch_size)

                # clip gradient self.x0ith large value 
                np.clip(grad, self.MIN_DLOSS, self.MAX_DLOSS, out = grad)

                # update policy
                self.x0 = self.x0 - lr * grad

                loss = self.loss_func.evaluate(X_batch, y_batch, self.x0)
                loss += self.regularizer.evaluate(self.x0, self.batch_size)

                error_history.append(loss)

            # avg_loss = np.mean(error_history)
            sum_loss = np.sum(error_history)

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
            
            if self.verbose:
                if iters % 2 == 0:
                    print("--Epoch = {}, average loss value = {}".format(str(iters), str(sum_loss/self.batch_size)))

        if not is_converged:
            raise ValueError("Not converged!")

        return self.opt_x
