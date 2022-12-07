import numpy as np

from .base import BaseOptimizer
from .utils import check_linesearch_params
from .line_search.backtracking import LineSearchBacktracking
from .line_search.bracketing import LineSearchBracketing
from .line_search.morethuente import LineSearchMorethuente

class LBFGS(BaseOptimizer):
    def __init__(self, x0,
        mem_size = 8, 
        tol = 1e-5, 
        delta = 1e-6, 
        past = 3, 
        max_iters = 0, 
        verbose = True,
        linesearch_params = None,
        loss_func = None, 
        linesearch_policy = "backtracking"):
            super(LBFGS, self).__init__(x0 = x0, 
                loss_func = loss_func,  
                max_iters = max_iters, 
                tol = tol,
                verbose = verbose)
            self.mem_size = mem_size
            self.delta = delta
            self.past = past
            self.linesearch_params = linesearch_params
            self.linesearch_policy = linesearch_policy

            check_linesearch_params(linesearch_params)
        
    def optimize(self, X, y):
        """
        Optimize the parameters of the model.
        """
        # the intial parameters to be optimized
        x = self.x0        
        num_dims = len(x)

        # intermediate variables
        xp = np.zeros((num_dims))
        g = np.zeros((num_dims))
        gp = np.zeros((num_dims))
        d = np.zeros((num_dims))
        # an array for storing previous values of the objective function
        pfx = np.zeros((max(1, self.past)))

        # define step search policy
        if self.linesearch_policy == "backtracking":
            linesearch = LineSearchBacktracking(X, y, self.loss_func, self.linesearch_params)
        elif self.linesearch_policy == "bracketing":
            linesearch = LineSearchBracketing(X, y, self.loss_func, self.linesearch_params)
        elif self.linesearch_params == "morethuente":
            linesearch = LineSearchMorethuente(X, y, self.loss_func, self.linesearch_params)
        else:
            raise ValueError("Cannot find line search policy.")

        # Initialize the limited memory
        mem_alpha = np.zeros((num_dims))
        mem_s = np.zeros((num_dims, self.mem_size))
        mem_y = np.zeros((num_dims, self.mem_size))
        mem_ys = np.zeros((self.mem_size))

        # Evaluate the function value and its gradient
        fx = self.loss_func.evaluate(X, y, x)
        g = self.loss_func.gradient(X, y, x)

        # Store the initial value of the cost function.
        pfx[0] = fx

        # Compute the direction we assume the initial hessian matrix H_0 as the identity matrix.
        d = -g

        # make sure the intial points are not sationary points (minimizer).
        xnorm = np.linalg.norm(x, ord = 2)
        gnorm = np.linalg.norm(g, ord = 2)

        if xnorm < 1.0: 
            xnorm = 1.0
        if (gnorm / xnorm) <= self.tol:
            print("ERROR: The initial variables already minimize the objective function.")
            return
        
        # compute intial step step = 1.0 / norm2(d)
        step = 1.0 / np.linalg.norm(d, ord = 2)
        
        k = 1
        end = 0
        # bound
        while(True):
            # store current x and gradient value
            xp = x.copy()
            gp = g.copy()

            # apply line search function to find optimized step, search for an optimal step
            # search(x, fx, g, d, step, xp)
            ls = linesearch.search(x, fx, g, d, step, xp)

            if ls["status"] < 0:
                x = xp.copy()
                g = gp.copy()
                print("ERROR: lbfgs exit: the point return to the privious point")
                return ls['status']

            fx = ls['fx']
            step = ls['step']
            x = ls['x']
            g = ls["g"]

            # Convergence test -- gradient
            # criterion is given by the following formula:
            # ||g(x)|| / max(1, ||x||) < tol
            xnorm = np.linalg.norm(x, ord = 2)
            gnorm = np.linalg.norm(g, ord = 2)

            # print the progress
            if self.verbose:
                print("Iteration = {}, fx = {}, xnorm value = {}, gnorm value = {}, ".format(k, fx, xnorm, gnorm))

            if xnorm < 1.0:
                xnorm = 1.0
            if (gnorm / xnorm) <= self.tol:
                print("INFO: success to reached convergence (g_epsilon).")
                break

            # Convergence test -- objective function value
            # The criterion is given by the following formula:
            # |f(past_x) - f(x)| / max(1, |f(x)|) < delta.
            if self.past <= k:
                rate = abs(pfx[k % self.past] - fx) / max(abs(fx), 1.0)
                if rate < self.delta:
                    print("INFO: success to meet stopping criteria (ftol)")
                    break
                # Store the current value of the cost function
                pfx[k % self.past] = fx

            if self.max_iters != 0 and self.max_iters < k + 1:
                print("INFO: the algorithm routine reaches the maximum number of iterations")
                break
            
            # Update vectors s and y:
            # s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
            # y_{k+1} = g_{k+1} - g_{k}.
            mem_s[:, end] = x - xp
            mem_y[:, end] = g - gp

            # Compute scalars ys and yy:
            # ys = y^t \cdot s = 1 / \rho.
            # yy = y^t \cdot y.
            # Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
            ys = np.dot(mem_y[:, end], mem_s[:, end])
            yy = np.dot(mem_y[:, end], mem_y[:, end])
            mem_ys[end] = ys

            # Compute the negative of gradients
            d = -g

            bound = self.mem_size if self.mem_size < k else k
            k += 1
            end = (end + 1) % self.mem_size

            j = end
            for i in range(bound):
                # if (--j == -1) j = m-1
                j = (j + self.mem_size - 1) % self.mem_size
                # \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}
                mem_alpha[j] = np.dot(mem_s[:, j], d) / mem_ys[j]
                # q_{i} = q_{i+1} - \alpha_{i} y_{i}
                d += (-mem_alpha[j]) * mem_y[:, j]

            d *= ys / yy

            for i in range(bound):
                beta = np.dot(mem_y[:, j], d) / mem_ys[j]
                d += (mem_alpha[j] - beta) *mem_s[:, j]
                j = (j + 1) % self.mem_size

            step = 1.0
        
        return x