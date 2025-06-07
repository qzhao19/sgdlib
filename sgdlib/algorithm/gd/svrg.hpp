#ifndef ALGORITHM_GD_SVRG_HPP_
#define ALGORITHM_GD_SVRG_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

/**
 * @file svrg.hpp
 *
 * @class SVRG
 *
 * @brief Implements the Stochastic Variance Reduced Gradient (SVRG) optimization algorithm.
 *
*/
class SVRG: public BaseOptimizer {
public:
    /**
     * @brief Constructor for the SVRG optimizer.
     *
     * Initializes the SVRG optimizer with the given parameters and passes them to the
     * base class `BaseOptimizer`.
     *
     * @param w0 Initial weight vector for the model.
     * @param loss The loss function to be minimized.
     * @param lr_policy The learning rate policy (e.g., constant, adaptive).
     * @param alpha L2 regularization parameter.
     * @param eta0 Initial learning rate.
     * @param tol Tolerance for convergence.
     * @param gamma Decay factor for the learning rate (used in some learning rate policies).
     * @param max_iters Maximum number of iterations for optimization.
     * @param num_inner Size of the mini-batch used for gradient computation.
     * @param random_seed Seed for the random number generator.
     * @param shuffle If true, shuffles the data before each epoch (default: true).
     * @param verbose If true, enables logging of optimization progress (default: true).
     *
     * @note This constructor calls the constructor of the base class `BaseOptimizer` to
     *       complete the initialization of the optimizer.
     * @see BaseOptimizer
     */
    SVRG(const std::vector<FeatValType>& w0,
        std::string loss,
        std::string lr_policy,
        FloatType alpha,
        FloatType eta0,
        FloatType tol,
        FloatType gamma,
        std::size_t max_iters,
        std::size_t num_inner,
        std::size_t random_seed,
        bool shuffle = true,
        bool verbose = true): BaseOptimizer(w0,
            loss,
            lr_policy,
            alpha,
            eta0,
            tol,
            gamma,
            max_iters,
            num_inner,
            random_seed,
            shuffle,
            verbose) {};

    /**
     * @brief Destructor for the SGD optimizer.
     *
     * Default destructor.
     */
    ~SVRG() = default;

    void optimize(const std::vector<FeatValType>& X,
                  const std::vector<LabelValType>& y) override {

        // Get the number of samples and features from the input data
        const std::size_t num_samples = y.size();
        const std::size_t num_features = this->w0_.size();
        const FeatValType inv_num_samples = 1.0 / static_cast<FeatValType>(num_samples);
        // Initialize iteration counter, sample index, and batch size
        std::size_t iter = 0, sample_index = 0;

        // Initialize convergence flag and best loss
        bool is_converged = false;
        bool is_infinity = false;
        FeatValType wscale = 1.0;

        const FeatValType eta_avg = this->eta0_ * inv_num_samples;
        const FeatValType eta_alpha = this->eta0_ * this->alpha_;

        //
        FeatValType reg_dw, fnorm, fnorm_ratio, init_fnorm, alpha_scaled;

        // initialize a lookup table for training X, y
        this->X_data_index_.resize(num_samples);
        std::iota(this->X_data_index_.begin(), this->X_data_index_.end(), 0);
        this->loss_history_.reserve(this->max_iters_ * this->num_inner_);

        // initialize gradient,
        FeatValType y_hat, grad, loss;
        std::vector<FeatValType> grad_history(num_samples, 0.0);
        std::vector<FeatValType> full_weight_update(num_features, 0.0);
        std::vector<std::size_t> update_history(num_features, 0);

        // initialize w0 (weight)
        std::vector<FeatValType> w0 = this->w0_;

        // start iteration
        for (iter = 0; iter < this->max_iters_; ++iter) {
            // Reset the full weight update vector to zero at the beginning of each iteration
            sgdlib::detail::vecset<FeatValType>(full_weight_update, 0.0);

#if defined(USE_OPENMP)
            int num_threads = 1;
            #pragma omp parallel
            {
                #pragma omp single
                num_threads = omp_get_max_threads();
            }
            // define independant local_grad for each thread
            std::vector<std::vector<FeatValType>> local_grad(
                num_threads, std::vector<FeatValType>(num_features, 0.0)
            );
            // OpenMP reduction for loss and manual reduction for grad
            #pragma omp parallel private(y_hat)
            {
                int thread_id = omp_get_thread_num();
                std::vector<FeatValType>& thread_grad = local_grad[thread_id];
                #pragma omp for nowait
                for (std::size_t i = 0; i < num_samples; ++i) {
                    y_hat = sgdlib::detail::vecdot<FeatValType>(
                        X.data() + (i * num_features),
                        X.data() + ((i + 1) * num_features),
                        w0.data()
                    );
                    grad_history[i] = this->loss_fn_->derivate(y_hat, y[i]);
                    sgdlib::detail::vecadd<FeatValType>(
                        X.data() + (i * num_features),
                        X.data() + ((i + 1) * num_features),
                        grad_history[i],
                        thread_grad
                    );
                }
            }
            // main thread reduction for grad
            for (std::size_t t = 0; t < num_threads; ++t) {
                sgdlib::detail::vecadd<FeatValType>(full_weight_update, local_grad[t], full_weight_update);
            }
#else
            // compute full gradeint for all data
            for (std::size_t i = 0; i < num_samples; ++i) {
                y_hat = sgdlib::detail::vecdot<FeatValType>(
                    X.data() + (i * num_features),
                    X.data() + ((i + 1) * num_features),
                    w0.data()
                );
                y_hat = y_hat * wscale;
                grad_history[i] = this->loss_fn_->derivate(y_hat, y[i]);
                sgdlib::detail::vecadd<FeatValType>(
                    X.data() + (i * num_features),
                    X.data() + ((i + 1) * num_features),
                    grad_history[i],
                    full_weight_update
                );
            }
#endif
            // inner loop
            for (std::size_t n = 0; n < this->num_inner_; ++n) {
                // shuffle the samples
                if (this->shuffle_) {
                    sample_index = this->random_state_.sample<std::size_t>(this->X_data_index_);
                }
                else {
                    sample_index = n % num_samples;
                }

                // just-in-time update for full weight 1/n * sum(d(f_k_w))
                // n - update_history[j]: weight is not be updated for n - update_history[j] times
                // need to compensate the full weight update
                // update_history[j] = n: weight is already updated now
                // Update the weights to compensate for the full gradient
                if (n > 0) {
                    for (std::size_t j = 0; j < num_features; ++j) {
                        w0[j] -= eta_avg / wscale * (n - update_history[j]) * full_weight_update[j];
                        update_history[j] = n;
                    }
                }

                // current gradient at sample_index row data
                y_hat = sgdlib::detail::vecdot<FeatValType>(
                    X.data() + (sample_index * num_features),
                    X.data() + ((sample_index + 1) * num_features),
                    w0.data()
                );
                y_hat = y_hat * wscale;
                loss = this->loss_fn_->evaluate(y_hat, y[sample_index]);
                grad = this->loss_fn_->derivate(y_hat, y[sample_index]);

                // update the weight scale factor based on eta*alpha
                wscale *= (1 - eta_alpha);

                // update w: old_gradient - new_gradient
                sgdlib::detail::vecadd<FeatValType>(
                    X.data() + (sample_index * num_features),
                    X.data() + ((sample_index + 1) * num_features),
                    this->eta0_ * (grad_history[sample_index] - grad) / wscale, w0
                );

                // check if the weight scale factor is below the threshold
                // rescale the weights and reset the scale factor
                if (wscale < WSCALE_THRESHOLD) {
                    sgdlib::detail::vecscale<FeatValType>(w0, wscale, w0);
                    wscale = 1.0;
                }

                // store loss value into this->loss_history_
                this->loss_history_.push_back(loss);
            }

            // finalize
            for (std::size_t j = 0; j < num_features; ++j) {
                w0[j] -= eta_avg / wscale * (this->num_inner_ - update_history[j]) * full_weight_update[j];
                update_history[j] = 0;
            }

            // compute the L2 norm of the loss function
            // fnorm = sqrt[(w0 * alpha + 1/n * d(f_k_w))**2]
            fnorm = 0.0;
            alpha_scaled = this->alpha_ * wscale;
            for (std::size_t j = 0; j < num_features; ++j) {
                reg_dw = full_weight_update[j] * inv_num_samples + alpha_scaled * w0[j];
                fnorm += reg_dw * reg_dw;
            }
            fnorm = std::sqrt(fnorm);

            // Store the initial gradient norm
            if (iter == 0) {
                init_fnorm = fnorm;
            }

            // Compute the ratio of the current gradient norm to the previous one
            fnorm_ratio = fnorm / init_fnorm;

            if (this->verbose_) {
                PRINT_RUNTIME_INFO(1, "Epoch = ", iter + 1, ", fnorm_ratio = ", fnorm_ratio);
            }

            if (fnorm_ratio <= this->tol_) {
                is_converged = true;
                break;
            }

            if (sgdlib::detail::hasinf<FeatValType>(w0)) {
                is_infinity = true;
                break;
            }
        }

        // shrink the this->loss_history_
        this->loss_history_.shrink_to_fit();

        if (callback_) {
            callback_(this->loss_history_);
        }

        if (is_infinity) {
            THROW_RUNTIME_ERROR("Floating-point under-/overflow occurred at epoch ", (iter + 1),
                                ", try to scale input data with standard or minmax.");
        }

        if (!is_converged) {
            THROW_RUNTIME_ERROR("Not converge, current number of epoch ", (iter + 1),
                                ", try apply different parameters.");
        }
        this->w_opt_ = w0;
    }

    const FeatValType get_intercept() const override {
        THROW_LOGIC_ERROR("The 'get_intercept' method is not supported for this SVRG optimizer.");
        return 0.0;
    }
};

}

#endif // ALGORITHM_GD_SVRG_HPP_
