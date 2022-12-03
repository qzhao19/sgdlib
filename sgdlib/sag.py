import numpy as np
from .base import BaseOptimizer

class SAG(BaseOptimizer):
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
            super(SAG, self).__init__(x0 = x0, 
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
        # self.x0 = np.ones((X.shape[1], 1))
        X_y = np.c_[X, y]
        is_converged = False
        no_improvement_count = 0
        
        num_batchs = X.shape[0] // self.batch_size
        _, num_features = X.shape

        grad_history = np.zeros((num_features, num_batchs))
        avg_grad = np.mean(grad_history, axis = 1)
        # avg_grad = avg_grad[np.neself.x0axis, :]

        for iters in range(self.max_iters):
            if self.shuffle:
                np.random.shuffle(X_y)
            error_history = []
            lr = self.lr_decay.compute(iters)
            for index in range(num_batchs):
                X_y_batch = X_y[self.batch_size * index : self.batch_size * (index + 1)]
                X_batch = X_y_batch[:, :-1]
                y_batch = X_y_batch[:, -1:]

                grad = self.loss_func.gradient(X_batch, y_batch, self.x0)
                grad = grad.squeeze()
                
                np.clip(grad, self.MIN_DLOSS, self.MAX_DLOSS, out = grad)

                avg_grad += (grad - grad_history[:, index]) / self.batch_size
                grad_history[:, index] = grad

                self.x0 = self.x0 - lr * avg_grad[:, np.newaxis]
                loss = self.loss_func.evaluate(X_batch, y_batch, self.x0)
                # print(loss)

                error_history.append(loss)
            
            sum_loss = np.sum(error_history)
            # avg_loss = np.mean(error_history)

            if self.tol > -np.Inf and sum_loss > best_loss - self.tol * self.batch_size:
            # if abs(best_loss - avg_loss) > self.tol:
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
                if iters % 1 == 0:
                    print("--Epoch = {}, average loss value = {}".format(str(iters), str(sum_loss/self.batch_size)))

        if not is_converged:
            raise ValueError("Not converged!")

        return self.opt_x

