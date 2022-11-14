import numpy as np
from line_search.backtracking import LineSearchBacktracking

class LBFGS(object):
    def __init__(self, x0,
        mem_size = 8, 
        tol = 1e-5, 
        delta = 1e-6, 
        cau_factor = 1.0e-6, 
        past = 3, 
        max_iters = 0, 
        max_linesearch = 64, 
        verbose = True,
        linesearch_params = None,
        loss_func = None, 
        linesearch_policy = "backtracking_linesearch"):
            self.x0 = x0,
            self.mem_size = mem_size
            self.tol = tol
            self.delta = delta
            self.cau_factor = cau_factor
            self.past = past
            self.max_iters = max_iters
            self.max_linesearch = max_linesearch
            self.verbose = verbose
            self.linesearch_params = linesearch_params
            self.loss_func = loss_func
            self.linesearch_policy = linesearch_policy

    def optimize(self, X, y):
        """
        Optimize the parameters of the model.
        """
        # the intial parameters to be optimized
        x = self.x0
        
        num_dims = x.shape[0]
        mem_size = self.mem_size

        # intermediate variables
        xp = np.zeros((num_dims))
        g = np.zeros((num_dims))
        gp = np.zeros((num_dims))
        d = np.zeros((num_dims))
        fx = np.zeros((max(1, self.past)))

        # Initialize the limited memory
        mem_alpha = np.zeros((num_dims))
        mem_s = np.zeros((num_dims, mem_size))
        mem_y = np.zeros((num_dims, mem_size))
        mem_ys = np.zeros((mem_size))

        # Evaluate the function value and its gradient
        fx0 = self.loss_func.evaluate(X, y, x)
        g = self.loss_func.gradient(X, y, x)

        # Store the initial value of the cost function.
        fx[0] = fx0

        # Compute the direction we assume the initial hessian matrix H_0 as the identity matrix.
        d = -g

        # define step search policy
        if self.linesearch_policy == "backtracking":
            linesearch = LineSearchBacktracking(X, y, self.loss_func, self.linesearch_params)
        else:
            raise ValueError("Cannot find line search policy.")

        # make sure the intial points are not sationary points (minimizer).
        xnorm = np.linalg.norm(x, ord = 2)
        gnorm = np.linalg.norm(g, ord = 2)

        if xnorm < 1.0: 
            xnorm = 1.0
        if (gnorm / max(1.0, xnorm)) <= self.tol:
            print("LBFGS_ALREADY_MINIMIZED")
            return
        
        #  compute intial step
        step = 1.0 / np.linalg.norm(d, ord = 2)
        
        num_iters = 1
        end = 0
        bound = 0

        while(True):
            # store current x and gradient value
            xp = x
            gp = g

            # min_step = self.min_step
            # max_step = self.max_step
            # apply line search function to find optimized step
            ls, x, fx0, g, d, step = linesearch.search(x, fx0, g, step, d, xp, gp)

            if ls < 0:
                x = xp
                g = gp
                print("lbfgs exit")
                break
            
            # print the progress
            if self.verbose:
                print("Iteration = {}, function value = {}".format(num_iters, fx0))

            # Convergence test -- gradient
            # criterion is given by the following formula:
            # ||g(x)|| / max(1, ||x||) < tol
            xnorm = np.linalg.norm(x, ord = 2)
            gnorm = np.linalg.norm(g, ord = 2)

            if (gnorm / max(1.0, xnorm)) <= self.tol:
                print("LBFGS_CONVERGENCE")
                break

            # Convergence test -- objective function value
            # The criterion is given by the following formula:
            # |f(past_x) - f(x)| / max(1, |f(x)|) < \delta.
            if self.past >= 0:
                rate = abs(fx[num_iters % self.past] - fx0) / max(1.0, abs(fx0))
                if rate < self.delta:
                    print("LBFGS_STOP")
                    break
                # Store the current value of the cost function
                fx[num_iters % self.past] = fx0

            if self.max_iters != 0 and self.max_iters <= num_iters:
                print("LBFGSERR_MAXIMUMITERATION")
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

            bound += 1
            bound = self.mem_size if self.mem_size < bound else bound
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
                beta = np.dot(mem_y[:, i], mem_ys[i])
                d += (mem_alpha[j] - beta) *mem_s[:, i]
                j = (j + 1) % self.mem_size

            step = 1.0
        
        return x





