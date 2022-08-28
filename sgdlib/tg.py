import numpy as np
MAX_DLOSS = 1e+10


class TruncatedGradient(object):
    def __init__(self,
        loss, 
        lr_decay, 
        regularizer, 
        tol = 0.001, 
        l1_ratio = 0.15,
        batch_size = 16, 
        max_iters = 2000, 
        num_iters_no_change = 5, 
        shuffle = True, 
        verbose = True):
            self.batch_size = batch_size
            self.max_iters = max_iters
            self.tol = tol
            self.l1_ratio = l1_ratio
            self.loss = loss
            self.lr_decay = lr_decay
            self.regularizer = regularizer
            self.num_iters_no_change = num_iters_no_change
            self.shuffle = shuffle
            self.verbose = verbose
            self.opt_w = None


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

        W = np.random.rand(X.shape[1], 1)
        # W = np.ones((X.shape[1], 1))

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

                grad = self.loss_fn.gradient(X_batch, y_batch, W)

                np.clip(grad, -MAX_DLOSS, MAX_DLOSS, out = grad)

                W = W - lr * grad
                
                max_cum_l1 += self.l1_ratio * lr * 0.0001
                W = self.update(W, cum_l1, max_cum_l1)

                loss = self.loss_fn.evaluate(X_batch, y_batch, W)
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
                self.opt_w = W
                break
                
            if iters % 1 == 0:
                print("--Epoch = {}, average loss value = {}".format(str(iters), str(sum_loss/self.batch_size)))

        if not is_converged:
            raise ValueError("Not converged!")

        return self.opt_w
        