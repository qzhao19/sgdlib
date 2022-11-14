import numpy as np
MAX_DLOSS = 1e+10


class SAG(object):
    def __init__(self,
        loss_func, 
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
            self.loss_func = loss_func
            self.lr_decay = lr_decay
            self.regularizer = regularizer
            self.num_iters_no_change = num_iters_no_change
            self.shuffle = shuffle
            self.verbose = verbose
            self.opt_w = None

    def optimize(self, X, y):
        best_loss = np.Inf
        
        W = np.random.rand(X.shape[1], 1)
        # W = np.ones((X.shape[1], 1))
        X_y = np.c_[X, y]

        is_converged = False
        no_improvement_count = 0
        
        num_batchs = X.shape[0] // self.batch_size
        num_samples, num_features = X.shape

        grad_history = np.zeros((num_features, num_batchs))
        avg_grad = np.mean(grad_history, axis = 1)
        # avg_grad = avg_grad[np.newaxis, :]

        for iters in range(self.max_iters):
            if self.shuffle:
                np.random.shuffle(X_y)
            error_history = []
            lr = self.lr_decay.compute(iters)
            for index in range(num_batchs):
                X_y_batch = X_y[self.batch_size * index : self.batch_size * (index + 1)]
                X_batch = X_y_batch[:, :-1]
                y_batch = X_y_batch[:, -1:]

                grad = self.loss_func.gradient(X_batch, y_batch, W)
                grad = grad.squeeze()
                
                np.clip(grad, -MAX_DLOSS, MAX_DLOSS, out = grad)

                avg_grad += (grad - grad_history[:, index]) / self.batch_size
                grad_history[:, index] = grad

                W = W - lr * avg_grad[:, np.newaxis]
                loss = self.loss_func.evaluate(X_batch, y_batch, W)
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
                self.opt_w = W
                break
            if self.verbose:
                if iters % 1 == 0:
                    print("--Epoch = {}, average loss value = {}".format(str(iters), str(sum_loss/self.batch_size)))

        if not is_converged:
            raise ValueError("Not converged!")

        return self.opt_w

