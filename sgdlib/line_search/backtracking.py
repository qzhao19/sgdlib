import numpy as np

class LineSearchBacktracking(object):
    def __init__(self, X, y, loss_func, linesearch_params):
        self.X = X
        self.y = y
        self.loss_func = loss_func
        self.linesearch_params = linesearch_params

    def search(self, x, fx, g, d, step, xp):

        result = {'status': 0, 'x': x, 'fx': fx, "g": g,'step': step} 

        # Decreasing and increasing factors
        decrease_factor = self.linesearch_params["decrease_factor"]
        increase_factor = self.linesearch_params['increase_factor']

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
        width = None
        dg = None
        count = 0
        while True:
            # x_{k+1} = x_k + step * d_k
            x = xp + step * d

            # fx = self.loss_fn.evaluate(x, g) 
            fx = self.loss_func.evaluate(self.X, self.y, x)
            g = self.loss_func.gradient(self.X, self.y, x)
            print("[INFO] Evaluate fx = %r step = %r." %(fx, step))

            # increment count
            count += 1

            if fx > fx_init + step * dg_test:
                print("[INFO] Not satisfy sufficient decrease condition.")
                width = decrease_factor
            else:
                # check Armijo condition
                if self.linesearch_params["condition"] == "LINESEARCH_BACKTRACKING_ARMIJO":
                    result = {'status': count, 'x': x, 'fx': fx, "g": g,'step': step}
                    return result

                # compute the project of d on the direction d
                dg = np.dot(g, d)
                if dg < self.linesearch_params["wolfe"] * dg_init:
                    print("[INFO] dg = %r < lbfgs_parameters.wolfe * dginit = %r" %(dg, self.linesearch_params["wolfe"] * dg_init))
                    print("[INFO]not satisfy wolf condition.")
                    width = increase_factor
                else:
                    # check wolf condition
                    if self.linesearch_params["condition"] == "LINESEARCH_BACKTRACKING_WOLFE":
                        result = {'status': count, 'x': x, 'fx': fx, "g": g,'step': step}
                        return result
                    
                    if dg > -self.linesearch_params["wolfe"] * dg_init:
                        width = decrease_factor
                    else:
                        result = {'status': count, 'x': x, 'fx': fx, "g": g,'step': step}
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
                result["status"] = -1
                return result
            
            step *= width