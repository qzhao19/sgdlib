import numpy as np

class BaseOptimizer(object):
    def __init__(self, 
        x0 = None, 
        loss_func = None, 
        lr_decay = None, 
        regularizer = None, 
        tol = 0.001, 
        batch_size = 16, 
        max_iters = 2000, 
        num_iters_no_change = 5, 
        shuffle = True, 
        verbose = True):
            self.x0 = x0
            self.loss_func = loss_func
            self.lr_decay = lr_decay
            self.regularizer = regularizer
            self.tol = tol
            self.batch_size = batch_size
            self.max_iters = max_iters
            self.num_iters_no_change = num_iters_no_change
            self.shuffle = shuffle
            self.verbose = verbose
            
            self.MAX_DLOSS = 1e+10
            self.MIN_DLOSS = -1e+10

            self.opt_x = None

    def check_params(self):
        pass
    
    