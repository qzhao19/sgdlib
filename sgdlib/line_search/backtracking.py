import numpy as np

class LineSearchBacktracking(object):
    def __init__(self, X, y, loss_func, linesearch_params):
        self.X = X
        self.y = y
        self.loss_func = loss_func
        self.linesearch_params = linesearch_params

    def search(self, x, fx0, g,  d, step, xp):
        # Decreasing and increasing factors
        decrease_factor = self.linesearch_params["decrease_factor"]
        increase_factor = self.linesearch_params['increase_factor']

        if step <= 0:
            print("'step' must be positive")
            return -1
        
        fx0_init = fx0

        # Compute the initial gradient in the search direction
        dg_init = np.dot(g, d)

        if dg_init > 0:
            print("Moving direction increases the objective function value")
            return -1
        
        dg_test = self.linesearch_params["ftol"] * dg_init
        width = None
        dg = None
        count = 0
        for _ in range(self.linesearch_params["max_linesearch"]):
            
            # x_{k+1} = x_k + step * d_k
            x = xp + step * d

            # fx0 = self.loss_fn.evaluate(x, g) 
            fx0 = self.loss_func.evaluate(self.X, self.y, x)
            g = self.loss_func.gradient(self.X, self.y, x)

            # increment count
            count += 1

            if fx0 > fx0_init + step * dg_test:
                width = decrease_factor
            else:
                # Armijo condition
                if self.linesearch_params["condition"] == "LINESEARCH_BACKTRACKING_ARMIJO":
                    return count

                # compute the project of d on the direction d
                dg = np.dot(g, d)
                if dg > self.linesearch_params["wolfe"] * dg_init:
                    width = increase_factor
                else:
                    # check wolf condition
                    if self.linesearch_params["condition"] == "LINESEARCH_BACKTRACKING_WOLFE":
                        return count
                    if dg > -self.linesearch_params["wolfe"] * dg_init:
                        width = decrease_factor
                    else:
                        return count

            if step < self.linesearch_params["min_step"]:
                print("the line search step became smaller than the minimum value allowed")
                return -1

            if step > self.linesearch_params["max_step"]:
                print("the line search step became larger than the maximum value allowed")
                return -1
        
            if count >= self.linesearch_params["max_linesearch"]:
                print("the line search step reached the max number of iterations")
                return -1
            
            step *= width
        
        return count, x, fx0, g, d, step