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
        width = None
        dg = None
        iter = 0
        for iter in range(self.linesearch_params.max_linesearch):
            x = xp + step * d

            # fx0 = self.loss_fn.evaluate(x, g) 
            fx0 = self.loss_func.evaluate(self.X, self.y, x)
            g = self.loss_func.gradient(self.X, self.y, x)

            if fx0 > fx0_init + step * dg_test:
                width = decrease_factor
            else:
                # Armijo condition
                if self.linesearch_params.condition == "LINESEARCH_BACKTRACKING_ARMIJO":
                    break

                # compute the project of d on the direction d
                dg = np.dot(g, d)
                if dg > self.linesearch_params.wolfe * dg_init:
                    width = increase_factor
                else:
                    # check wolf condition
                    if self.linesearch_params.condition == "LINESEARCH_BACKTRACKING_WOLFE":
                        break
                    if dg > -self.linesearch_params.wolfe * dg_init:
                        width = decrease_factor
                    else:
                        break

            if step < self.linesearch_params.min_step:
                raise ValueError("the line search step became smaller than the minimum value allowed")

            if step > self.linesearch_params.max_step:
                raise ValueError("the line search step became larger than the maximum value allowed")

            step *= width
        
        if iter >= self.linesearch_params.max_linesearch:
            raise ValueError("the line search step reached the max number of iterations")