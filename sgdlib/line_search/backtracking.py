import numpy as np

class LineSearchBacktracking(object):
    def __init__(self, X, y, loss_fn, linesearch_params):
        self.X = X
        self.y = y
        self.loss_fn = loss_fn
        self.linesearch_params = linesearch_params

    def search(self, x, fx0, g, step, d, xp, gp):

        # Decreasing and increasing factors
        decrease_factor = self.linesearch_params.decrease_factor
        increase_factor = self.linesearch_params.increase_factor

        if step <= 0:
            raise ValueError("'step' must be positive")
        
        fx0_init = fx0

        # Compute the initial gradient in the search direction
        dg_init = np.dot(g, d)

        if dg_init > 0:
            raise RuntimeError("Moving direction increases the objective function value")
        
        dg_test = self.linesearch_params.ftol * dg_init
        width = 0

        iter = 0
        for iter in range(self.linesearch_params.max_linesearch):
            x = xp + step * d

            # fx0 = self.loss_fn.evaluate(x, g) 
            fx0 = self.loss_func.evaluate(self.X, self.y, x)
            g = self.loss_func.gradient(self.X, self.y, x)


            