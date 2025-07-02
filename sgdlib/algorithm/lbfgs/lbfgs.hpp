#ifndef ALGORITHM_LBFGS_LBFGS_HPP_
#define ALGORITHM_LBFGS_LBFGS_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

class LBFGS: public Optimizer {
private:
    struct LimitedMemoryData {
        sgdlib::FeatureScalarType mem_ys;
        sgdlib::FeatureScalarType mem_alpha;
        std::vector<sgdlib::FeatureScalarType> mem_y;
        std::vector<sgdlib::FeatureScalarType> mem_s;
    };
    using LimitedMemoryDataType = LimitedMemoryData;

public:
    LBFGS(const std::vector<sgdlib::FeatureScalarType>& w0,
          std::string loss,
          std::string search_policy,
          sgdlib::ScalarType delta,
          sgdlib::ScalarType tol,
          std::size_t max_iters,
          std::size_t mem_size,
          std::size_t past,
          std::shared_ptr<sgdlib::StepSizeSearchParamType> stepsize_search_params,
          bool verbose = true): Optimizer(w0, loss, tol, max_iters, verbose) {
        this->search_policy_ = search_policy;
        this->delta_ = delta;
        this->mem_size_ = mem_size;
        this->past_ = past;
        if (stepsize_search_params == nullptr) {
            this->stepsize_search_params_ = std::make_shared<sgdlib::StepSizeSearchParamType>(
                sgdlib::detail::DEFAULT_STEPSIZE_SEARCH_PARAMS
            );
            PRINT_RUNTIME_INFO(1, "Use default search parameters.");
        }
        this->stepsize_search_params_ = stepsize_search_params;
        init_loss_params();
    };

    ~LBFGS() = default;

    void optimize(const sgdlib::ArrayDatasetType& dataset) override {

        const std::size_t num_samples = dataset.nrows();
        const std::size_t num_features = dataset.ncols();
        const sgdlib::FeatureScalarType inv_num_samples =1.0 / static_cast<sgdlib::FeatureScalarType>(num_samples);

        // initialize w0 (weight)
        std::vector<sgdlib::FeatureScalarType> x = this->w0_;

        // define the initial parameters
        sgdlib::FeatureScalarType y_hat;
        std::size_t i, j, k, end, bound;
        sgdlib::FeatureScalarType rate, beta = 0.0;
        sgdlib::FeatureScalarType fx = 0.0, ys = 0.0, yy = 0.0;

        // init the loss history vector
        this->loss_history_.reserve(num_samples * this->max_iters_);

        // intermediate variables: previous x, gradient, previous gradient, directions
        std::vector<sgdlib::FeatureScalarType> xp(num_features);
        std::vector<sgdlib::FeatureScalarType> g(num_features);
        std::vector<sgdlib::FeatureScalarType> gp(num_features);
        std::vector<sgdlib::FeatureScalarType> d(num_features);

        // initialize the limited memory data
        // mem_s: storing changes of parameters in the past
        // mem_y: storing changes of gradient in the past
        // mem_ys: storing value of y_T_k @ s_k
        std::vector<LimitedMemoryDataType> limited_mem_data(this->mem_size_);
        for (std::size_t i = 0; i < this->mem_size_; ++i) {
            limited_mem_data[i].mem_y.resize(num_features);
            limited_mem_data[i].mem_s.resize(num_features);
            limited_mem_data[i].mem_ys = 0.0;
            limited_mem_data[i].mem_alpha = 0.0;
        }

        // vector for storing previous values of the objective function
        std::vector<sgdlib::FeatureScalarType> pfx(std::max(static_cast<std::size_t>(1), this->past_));

        // call step search policy
        std::unique_ptr<sgdlib::detail::StepSizeSearch> stepsize_search;
        if (this->search_policy_ == "BacktrackingLineSearch") {
            stepsize_search = std::make_unique<sgdlib::detail::BacktrackingLineSearch>(
                dataset, this->loss_fn_, this->stepsize_search_params_
            );
        }
        else if (this->search_policy_ == "BracketingLineSearch") {
            stepsize_search = std::make_unique<sgdlib::detail::BracketingLineSearch>(
                dataset, this->loss_fn_, this->stepsize_search_params_
            );
        }
        else {
            THROW_INVALID_ERROR("LBFGS optimizer supports 'backtracking' or 'bracketing' policy only.");
        }

        // compute intial loss value and gradeint with full data X
        fx = this->loss_fn_->evaluate_with_gradient(dataset, x, g);
        fx *= inv_num_samples;
        sgdlib::detail::vecscale<sgdlib::FeatureScalarType>(g, inv_num_samples, g);
        // store the initial value of the cost function fx
        pfx[0] = fx;

        // compute the direction, initial hessian matrix H_0 as the identity matrix
        for (std::size_t n = 0; n < num_features; ++n) {
            d[n] = -g[n];
        }

        // compute norm2 of g and d to make sure initial vars are not stationary point
        sgdlib::FeatureScalarType xnorm = sgdlib::detail::vecnorm2<sgdlib::FeatureScalarType>(x, false);
        sgdlib::FeatureScalarType gnorm = sgdlib::detail::vecnorm2<sgdlib::FeatureScalarType>(g, false);

        // Convergence test 0 -- gradient
        // make sure that the initial variables are not a minimizer
        // ||g(x)|| / max(1, ||x||) < tol
        if (gnorm / std::max(1.0, xnorm) <= this->tol_) {
            THROW_RUNTIME_ERROR("initial variables already are stationary point.");
        }

        // intial step size stepsize = 1.0 / norm2(d)
        sgdlib::FeatureScalarType stepsize = 1.0 / sgdlib::detail::vecnorm2<sgdlib::FeatureScalarType>(d, false);

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

            // restore loss fx into loss_history
            this->loss_history_.push_back(fx);

            // Convergence test 1 -- gradient test
            // criterion is given by the following formula:
            // ||g(x)|| / max(1, ||x||) < tol
            xnorm = sgdlib::detail::vecnorm2<sgdlib::FeatureScalarType>(x, false);
            gnorm = sgdlib::detail::vecnorm2<sgdlib::FeatureScalarType>(g, false);

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
            sgdlib::detail::vecdiff<sgdlib::FeatureScalarType>(x, xp, limited_mem_data[end].mem_s);
            sgdlib::detail::vecdiff<sgdlib::FeatureScalarType>(g, gp, limited_mem_data[end].mem_y);

            // Compute scalars ys and yy:
            // ys = y^t @ s, s = 1 / rho.
            // yy = y^t @ y.
            // Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
            ys = sgdlib::detail::vecdot<sgdlib::FeatureScalarType>(limited_mem_data[end].mem_y, limited_mem_data[end].mem_s);
            yy = sgdlib::detail::vecdot<sgdlib::FeatureScalarType>(limited_mem_data[end].mem_y, limited_mem_data[end].mem_y);
            limited_mem_data[end].mem_ys = ys;

            // compute negative of gradient: d = -g
            sgdlib::detail::vecncpy<sgdlib::FeatureScalarType>(g, d);

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
                limited_mem_data[j].mem_alpha = sgdlib::detail::vecdot<sgdlib::FeatureScalarType>(
                    limited_mem_data[j].mem_s, d
                );
                limited_mem_data[j].mem_alpha /= limited_mem_data[j].mem_ys;

                // update d_{i} = d_{i+1} - (alpha_{i} * y_{i})
                sgdlib::detail::vecadd<sgdlib::FeatureScalarType>(
                    limited_mem_data[j].mem_y,
                    -limited_mem_data[j].mem_alpha, d
                );
            }

            // scale Hessian H_0, dirctyion *= scale_ceoff;
            const sgdlib::FeatureScalarType scale_ceoff = ys / yy;
            sgdlib::detail::vecscale<sgdlib::FeatureScalarType>(d, scale_ceoff, d);

            // loop2: backwards recursion
            for (i = 0; i < bound; ++i) {
                // compute beta_j = rho_{j} * y_{T}_{j} @ d_{J}, rho_{j} = 1/mem_ys
                beta = sgdlib::detail::vecdot<sgdlib::FeatureScalarType>(
                    limited_mem_data[j].mem_y, d
                );
                beta /= limited_mem_data[j].mem_ys;

                // update gamm_{i+1} = gamm_{i} + (alpha_{j} - beta_{j}) * s_{j}
                sgdlib::detail::vecadd<sgdlib::FeatureScalarType>(
                    limited_mem_data[j].mem_s,
                    limited_mem_data[j].mem_alpha - beta,
                    d
                );

                // starting the earliest history information to traverse backward
                j = (j + 1) % this->mem_size_;
            }

            ys = 0.0;
            yy = 0.0;
            stepsize = 1.0;
        }
        this->loss_history_.shrink_to_fit();

        if (callback_) {
            callback_(this->loss_history_);
        }

        this->w_opt_ = x;
    };

    const sgdlib::FeatureScalarType get_intercept() const override {
        THROW_LOGIC_ERROR("The 'get_intercept' method is not supported for this LBFGS optimizer.");
        return 0.0;
    }
};

} // namespace sgdlib

#endif // ALGORITHM_LBFGS_LBFGS_HPP_
