#ifndef ALGORITHM_LBFGS_LBFGS_HPP_
#define ALGORITHM_LBFGS_LBFGS_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

/**
 * @file lbfgs.hpp
 * 
 * @class LBFGS
 * 
 * @brief Implements the Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) 
 * optimization algorithm.
 *
 * This class inherits from `BaseOptimizer` and provides functionality for optimizing
 * machine learning models using the L-BFGS algorithm. L-BFGS is a quasi-Newton method
 * that approximates the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm using a limited
 * amount of memory, making it suitable for large-scale optimization problems.
 */
class LBFGS: public BaseOptimizer {
public:
    /**
     * @brief Constructor for the LBFGS optimizer.
     *
     * Initializes the LBFGS optimizer with the given parameters and passes them to the
     * base class `BaseOptimizer`.
     *
     * @param w0 Initial weight vector for the model.
     * @param loss Name of the loss function, of type `std::string`. 
     *             Supported loss functions include "LogLoss", "hinge_loss", etc.
     * @param search_policy Name of the search policy for selecting search directions during optimization. 
     *                      Supported policies include "ARMIJO", "WOLFE", etc.
     * @param delta The parameter for convergence test, it determines the minimum rate of 
     *              decrease of theobjective function
     * @param tol Tolerance for convergence. Optimization stops when the change 
     *            in the objective function value is less than this threshold.
     * @param max_iters Maximum number of iterations for optimization.
     * @param mem_size Memory size，the number of corrections to approximate the inverse hessian matrix.
     * @param past Number of past iterations to consider for estimation.
     * @param stepsize_search_params Step size search parameters, of type `StepSizeSearchParamType*`. 
     *                               Contains specific parameters for step size search (e.g. dec_factor).
     * @param shuffle If true, shuffles the data before each epoch (default: true).
     * @param verbose If true, enables logging of optimization progress (default: true).
     * 
     * @note This constructor calls the constructor of the base class `BaseOptimizer` to 
     *       complete the initialization of the optimizer.
     * @see BaseOptimizer
     */
    LBFGS(const std::vector<FeatValType>& w0, 
          std::string loss, 
          std::string search_policy,
          FloatType delta,
          FloatType tol,
          std::size_t max_iters, 
          std::size_t mem_size,
          std::size_t past,
          StepSizeSearchParamType* stepsize_search_params,
          bool shuffle = true, 
          bool verbose = true): BaseOptimizer(w0,
            loss, 
            search_policy,
            delta,
            tol, 
            max_iters, 
            mem_size,
            past,
            stepsize_search_params,
            shuffle, 
            verbose) {};
    
    ~LBFGS() = default;

    void optimize(const std::vector<FeatValType>& X, 
                  const std::vector<LabelValType>& y) override {

        std::size_t num_samples = y.size();
        std::size_t num_features = this->w0_.size();

        // initialize w0 (weight)
        std::vector<FeatValType> x = this->w0_;

        // define the initial parameters
        FeatValType y_hat;
        std::size_t i, j, k, end, bound;
        FloatType rate, beta;
        FeatValType fx = 0.0, ys = 0.0, yy = 0.0;
        
        // intermediate variables: previous x, gradient, previous gradient, directions
        std::vector<FeatValType> xp(num_features);
        std::vector<FeatValType> g(num_features);
        std::vector<FeatValType> gp(num_features);
        std::vector<FeatValType> d(num_features);
        
        // initialize the limited memory variables
        // mem_s: storing changes of parameters in the past
        // mem_y: storing changes of gradient in the past
        // mem_ys: storing value of y_T_k @ s_k
        std::vector<FeatValType> mem_s(num_features * this->mem_size_, 0.0);
        std::vector<FeatValType> mem_y(num_features * this->mem_size_, 0.0);
        std::vector<FeatValType> mem_ys(this->mem_size_, 0.0);
        std::vector<FeatValType> mem_alpha(this->mem_size_, 0.0);

        // vector for storing previous values of the objective function
        std::vector<FeatValType> pfx(std::max(static_cast<std::size_t>(1), this->past_));

        // call step search policy
        std::unique_ptr<sgdlib::StepSizeSearch<sgdlib::LossFunction>> stepsize_search;
        if (this->search_policy_ == "backtracking") {
            stepsize_search = std::make_unique<sgdlib::BacktrackingLineSearch<sgdlib::LossFunction>>(
                X, y, this->loss_fn_, this->stepsize_search_params_
            );
        }
        else if (this->search_policy_ == "bracketing") {
            stepsize_search = std::make_unique<sgdlib::BracketingLineSearch<sgdlib::LossFunction>>(
                X, y, this->loss_fn_, this->stepsize_search_params_
            );
        }
        else {
            THROW_INVALID_ERROR("LBFGS optimizer supports 'backtracking' or 'bracketing' policy only.");
        }

        // compute intial loss value and gradeint
        for (std::size_t n = 0; n < num_samples; ++n) {
            y_hat = std::inner_product(&X[n * num_features], 
                                       &X[(n + 1) * num_features], 
                                       x.begin(), 0.0);
            fx += this->loss_fn_->evaluate(y_hat, y[n]);
            for (std::size_t m = 0; m < num_features; ++m) {
                g[m] += this->loss_fn_->derivate(y_hat, y[n]) * X[n * num_features + m];
            }
        }
        fx /= static_cast<FeatValType>(num_samples);
        std::transform(g.begin(), g.end(), g.begin(),
                      [num_samples](FeatValType val) { 
                        return val / static_cast<FeatValType>(num_samples); 
                      });
        
        // store the initial value of the cost function fx
        pfx[0] = fx;

        // compute the direction, initial hessian matrix H_0 as the identity matrix
        for (std::size_t n = 0; n < num_features; ++n) {
            d[n] = -g[n];
        }

        // compute norm2 of g and d to make sure initial vars are not stationary point
        FeatValType xnorm = sgdlib::internal::sqnorm2<FeatValType>(x, true);
        FeatValType gnorm = sgdlib::internal::sqnorm2<FeatValType>(g, true);

        // Convergence test 0 -- gradient
        // make sure that the initial variables are not a minimizer
        // ||g(x)|| / max(1, ||x||) < tol
        if (gnorm / std::max(1.0, xnorm) <= this->tol_) {
            THROW_RUNTIME_ERROR("initial variables already are stationary point.");
        }

        // intial step size stepsize = 1.0 / norm2(d)
        FeatValType stepsize = 1.0 / sgdlib::internal::sqnorm2<FeatValType>(d, true);

        k = 1;
        end = 0;
        bound = 0;
        for (;;) {
            // store current xp = x and gp = g
            std::copy(x.begin(), x.end(), xp.begin());
            std::copy(g.begin(), g.end(), gp.begin());
            
            // call step size search function
            int search_status = stepsize_search->search(xp, gp, d, x, g, fx, stepsize);
            if (search_status < 0) {
                // revert to previous point
                std::copy(xp.begin(), xp.end(), x.begin());
                std::copy(gp.begin(), gp.end(), g.begin());
                THROW_RUNTIME_ERROR("lbfgs exit, the point return to the privious point.");
            }

            // Convergence test 1 -- gradient test
            // criterion is given by the following formula:
            // ||g(x)|| / max(1, ||x||) < tol
            xnorm = sgdlib::internal::sqnorm2<FeatValType>(x, true);
            gnorm = sgdlib::internal::sqnorm2<FeatValType>(g, true);
            
            if (this->verbose_) {
                PRINT_RUNTIME_INFO(1, "iteration = ", k, 
                                   ", loss = ", fx, 
                                   ", xnorm value = ", xnorm, 
                                   ", gnorm value = ", gnorm);
            }
            if (gnorm / std::max(1.0, xnorm) <= this->tol_) {
                PRINT_RUNTIME_INFO(1, "success to reached convergence (tol).");
                break;
            }

            // Convergence test 2 -- objective function value
            // The criterion is given by the following formula:
            // |f(past_x) - f(x)| / max(1, |f(x)|) < delta.
            if (this->past_ <= k) {
                // compute the relative improvement from the past.
                rate = std::abs(pfx[k % this->past_] - fx) / std::max(std::abs(fx), 1.0);
                if (rate < this->delta_) {
                    PRINT_RUNTIME_INFO(1, "success to meet stopping criteria (ftol).");
                    break;
                }
                pfx[k % this->past_] = fx;
            }
            if ((this->max_iters_ != 0) && (this->max_iters_ < k + 1)) {
                PRINT_RUNTIME_INFO(1, "algorithm routine reaches the maximum number of iterations");
                break;
            }
            
            // Update vectors s and y:
            // s_{k+1} = x_{k+1} - x_{k} = step * d_{k}.
            // y_{k+1} = g_{k+1} - g_{k}.
            for (std::size_t n = 0; n < num_features; ++n) {
                mem_s[n * this->mem_size_ + end] = x[n] - xp[n];
                mem_y[n * this->mem_size_ + end] = g[n] - gp[n];
            }

            // Compute scalars ys and yy:
            // ys = y^t @ s, s = 1 / rho.
            // yy = y^t @ y.
            // Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
            for (std::size_t n = 0; n < num_features; ++n) {
                ys += mem_y[n * this->mem_size_ + end] * mem_s[n * this->mem_size_ + end];
                yy += mem_y[n * this->mem_size_ + end] * mem_y[n * this->mem_size_ + end];
            }
            mem_ys[end] = ys;
            
            // compute negative of gradient: d = -g
            for (std::size_t n = 0; n < num_features; ++n) {
                d[n] = -g[n];
            }

            // bound: number of currently available historical messages 
            // k: number of iterations
            // end: indicates the location of the latest history information.
            //      after each iteration, end is updated to the next position
            // j: index for traversing history information
            bound = (this->mem_size_ <= k) ? this->mem_size_ : k;
            ++k;
            end = (end + 1) % this->mem_size_;
            j = end;

            // loop1: forwards recursion
            for (i = 0; i < bound; ++i) {
                // if (--j == -1) j = m-1 traverse history forward, 
                // starting with the most recent history message
                j = (j + this->mem_size_ - 1) % this->mem_size_;
                // alpha_{j} = s^{T}_{j} @ d_{j} * rho_{j}, rho_{j} = 1/mem_ys
                FeatValType ds = 0.0;
                for (std::size_t n = 0; n < num_features; ++n) {
                    ds += mem_s[n * this->mem_size_ + j] * d[n];
                }
                mem_alpha[j] = ds / mem_ys[j];
                // update d_{i} = d_{i+1} - (alpha_{i} * y_{i})
                for (std::size_t n = 0; n < num_features; ++n) {
                    d[n] += (-mem_alpha[j]) * mem_y[n * this->mem_size_ + j];
                }
            }

            // scale Hessian H_0
            const FeatValType scale_ceoff = ys / yy;
            for (std::size_t n = 0; n < num_features; ++n) {
                d[n] *= scale_ceoff;
            }
            
            // loop2: backwards recursion
            for (i = 0; i < bound; ++i) {
                // compute beta_j = rho_{j} * y_{T}_{j} @ d_{J}, rho_{j} = 1/mem_ys
                FeatValType yd = 0.0;
                for (std::size_t n = 0; n < num_features; ++n) {
                    yd += mem_y[n * this->mem_size_ + j] * d[n];
                }
                beta = yd / mem_ys[j];
                // update gamm_{i+1} = gamm_{i} + (alpha_{j} - beta_{j}) s_{j}
                for (std::size_t n = 0; n < num_features; ++n) {
                    d[n] += (mem_alpha[j] - beta) * mem_s[n * this->mem_size_ + j];
                }
                // starting the earliest history information to traverse backward 
                j = (j + 1) % this->mem_size_;
            }
            
            ys = 0.0;
            yy = 0.0;
            stepsize = 1.0;
        }
        this->w_opt_ = x;
    };

    const FeatValType get_intercept() const override {
        THROW_RUNTIME_ERROR("The 'get_intercept' method is not supported for this LBFGS optimizer.");
        return 0.0;
    }
};

} // namespace sgdlib

#endif // ALGORITHM_LBFGS_LBFGS_HPP_