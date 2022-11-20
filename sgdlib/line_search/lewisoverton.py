import numpy as np

class LineSearchLewisoverton(object):
    def __init__(self, X, y, loss_func, linesearch_params):
        self.X = X
        self.y = y
        self.loss_func = loss_func
        self.linesearch_params = linesearch_params

    def search(self, x, fx0, g,  d, step, xp):
        
        if step <= 0:
            print("'step' must be positive")
            return -1
        
        fx0_init = fx0

        # Compute the initial gradient in the search direction
        dg_init = np.dot(g, d)

        if dg_init > 0:
            print("Moving direction increases the objective function value")
            return -1
        
        