import numpy as np

MAX_DLOSS = 1e+10

class SGD(object):
    """Mini-batch stochastic gradient descent algorithm

    Parameters
    ----------
    loss : class object
        _description_
    """
    def __init__(self, 
        loss, 
        lr_decay, 
        regularizer, 
        tol = 0.001, 
        batch_size = 16, 
        max_iters = 2000, 
        num_iters_no_change = 5, 
        shuffle = True, 
        verbose = True):
            self.batch_size = batch_size
            self.max_iters = max_iters
            self.tol = tol
            self.loss = loss
            self.lr_decay = lr_decay
            self.regularizer = regularizer
            self.num_iters_no_change = num_iters_no_change
            self.shuffle = shuffle
            self.verbose = verbose
            self.opt_w = None

    def optimize(self, X, y):
        best_loss = np.Inf
        W = np.random.rand(X.shape[1], 1)
        
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

                grad = self.loss.gradient(X_batch, y_batch, W)
                grad += self.regularizer.gradient(W, self.batch_size)

                # clip gradient with large value 
                np.clip(grad, -MAX_DLOSS, MAX_DLOSS, out = grad)

                # update policy
                W = W - lr * grad

                loss = self.loss.evaluate(X_batch, y_batch, W)
                loss += self.regularizer.evaluate(W, self.batch_size)

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
                self.opt_w = W
                break
            
            if self.verbose:
                if iters % 2 == 0:
                    print("--Epoch = {}, average loss value = {}".format(str(iters), str(sum_loss/self.batch_size)))

        if not is_converged:
            raise ValueError("Not converged!")

        return self.opt_w
