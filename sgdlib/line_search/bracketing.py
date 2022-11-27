import numpy as np

class LineSearchBracketing(object):
    def __init__(self, X, y, loss_func, linesearch_params):
        self.X = X
        self.y = y
        self.loss_func = loss_func
        self.linesearch_params = linesearch_params

    def search(self, x, fx, g, d, step, xp):

        result = {'status': 0, 'x': x, 'fx': fx, "g": g, 'step': step} 
        
        if step <= 0:
            print("'step' must be positive")
            result["status"] = -1
            return result
        
        fx_init = fx
        # Compute the initial gradient in the search direction
        dg_init = np.dot(g, d)

        if dg_init > 0:
            print("Moving direction increases the objective function value")
            result["status"] = -1
            return result
        
        dg_test = self.linesearch_params["ftol"] * dg_init
        step_lo = 0
        step_hi = np.inf
        dg = None
        count = 0
        while True:
            # x_{k+1} = x_k + step * d_k
            x = xp + step * d

            fx = self.loss_func.evaluate(self.X, self.y, x)
            g = self.loss_func.gradient(self.X, self.y, x)

            # increment count
            count += 1

            if fx > fx_init + step * dg_test:
                step_hi = step
            else:
                # check Armijo condition
                if self.linesearch_params["condition"] == "ARMIJO":
                    result = {'status': count, 'x': x, 'fx': fx, "g": g,'step': step}
                    return result

                # compute the project of d on the direction d
                dg = np.dot(g, d)
                if dg < self.linesearch_params["wolfe"] * dg_init:
                    step_lo = step
                else:
                    # check wolf condition
                    if self.linesearch_params["condition"] == "WOLFE":
                        result = {'status': count, 'x': x, 'fx': fx, "g": g, 'step': step}
                        return result
                    
                    if dg > -self.linesearch_params["wolfe"] * dg_init:
                        step_hi = step
                    else:
                        result = {'status': count, 'x': x, 'fx': fx, "g": g, 'step': step}
                        return result

            if step < self.linesearch_params["min_step"]:
                print("the line search step became smaller than the minimum value allowed")
                result["status"] = -1
                return result

            if step > self.linesearch_params["max_step"]:
                print("the line search step became larger than the maximum value allowed")
                result["status"] = -1
                return result
        
            if count >= self.linesearch_params["max_linesearch"]:
                print("the line search step reached the max number of iterations")
                result = {'status': 0, 'x': x, 'fx': fx, "g": g, 'step': step}
                return result
            
            if step_hi == np.inf:
                step = step * 2.0
            else:
                step = (step_lo + step_hi) / 2.0