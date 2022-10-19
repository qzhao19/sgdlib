import numpy as np

class LBFGS(object):
    def __init__(self, x0,
        mem_size = 8, 
        tol = 1e-5, 
        delta = 1e-6, 
        cau_factor = 1.0e-6, 
        past = 3, 
        max_iters = 0, 
        max_linesearch = 64, 
        max_step = 1.0e+20, 
        min_step = 1.0e-20, 
        verbose = True,
        loss = None,
        line_search_policy = None):
            self.x0 = x0,
            self.mem_size = mem_size
            self.tol = tol
            self.delta = delta
            self.cau_factor = cau_factor
            self.past = past
            self.max_iters = max_iters
            self.max_linesearch = max_linesearch
            self.max_step = max_step
            self.min_step = min_step
            self.verbose = verbose
            self.loss = loss
            self.line_search_policy = line_search_policy

    def optimize(self, X, y):
        """
        Optimize the parameters of the model.
        """
        num_dims = self.x0.shape[0]
        mem_size = self.mem_size

        # intermediate variables
        prev_x = np.zeros((num_dims))
        grad = np.zeros((num_dims))
        prev_grad = np.zeros((num_dims))
        diection = np.zeros((num_dims))
        losses = np.zeros((max(1, self.past)))

        # Initialize the limited memory
        mem_alpha = np.zeros((num_dims))
        mem_s = np.zeros((num_dims, mem_size))
        mem_y = np.zeros((num_dims, mem_size))
        mem_ys = np.zeros((mem_size))

        # Evaluate the function value and its gradient
        loss = self.loss.evaluate(X, y, self.x0)
        grad = self.loss.gradient(X, y, self.x0)

        # Store the initial value of the cost function.
        losses[0] = loss

        # Compute the direction we assume the initial hessian matrix H_0 as the identity matrix.
        diection = -grad
        















