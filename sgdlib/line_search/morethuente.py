import numpy as np

class LineSearchMorethuente(object):
    def __init__(self, X, y, loss_func, linesearch_params):
        self.X = X
        self.y = y
        self.loss_func = loss_func
        self.linesearch_params = linesearch_params

    def search(self, x, fx, g,  d, step, xp):
        
        if step <= 0:
            print("'step' must be positive")
            return -1
        
        fx_init = fx

        # Compute the initial gradient in the search direction
        dg_init = np.dot(g, d)

        if dg_init > 0:
            print("Moving direction increases the objective function value")
            return -1
        
        