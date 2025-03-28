#ifndef ALGORITHM_SGD_SVRG_HPP_
#define ALGORITHM_SGD_SVRG_HPP_

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
            alpha, eta0, 
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
        std::size_t num_samples = y.size();
        std::size_t num_features = this->w0_.size();
        // Initialize iteration counter, sample index, and batch size
        std::size_t iter = 0, sample_index = 0;
        std::size_t batch_size = num_samples / this->num_inner_;

        // initialize a lookup table for training X, y
        std::vector<std::size_t> X_data_index(num_samples);
        std::iota(X_data_index.begin(), X_data_index.end(), 0);
        
        bool is_converged = false;
        bool is_infinity = false;
        FloatType best_loss = INF;
        FeatValType wscale = 1.0;
        
        FloatType tmp, fnorm, fnorm_ratio, prev_fnorm;

        // initialize a lookup table for training X, y
        std::vector<std::size_t> X_data_index(num_samples);
        std::iota(X_data_index.begin(), X_data_index.end(), 0);

        // initialize gradient, 
        FeatValType y_hat, grad;
        std::vector<FeatValType> grad_history(num_samples, 0.0);
        std::vector<FeatValType> full_weight_update(num_features, 0.0);
        std::vector<std::size_t> update_history(num_features, 0);

        // initialize w0 (weight)
        std::vector<FeatValType> w0 = this->w0_;

        for (iter = 0; iter < this->max_iters_; ++iter) {
            // compute full gradeint for all data
            // Reset the full weight update vector to zero at the beginning of each iteration
            std::memset(full_weight_update.data(), 0, full_weight_update.size() * sizeof(FeatValType));
            // Compute the full gradient for all samples
            for (std::size_t i = 0; i < num_samples; ++i) {
                y_hat = std::inner_product(&X[i * num_features], 
                                           &X[(i + 1) * num_features], 
                                           w0.begin(), 0.0);
                y_hat = y_hat * wscale;
                grad_history[i] = this->loss_fn_->derivate(y_hat, y[i]);
                for (std::size_t j = 0; j < num_features; ++j) {
                    full_weight_update[j] += grad_history[i] * X[i * num_features + j];
                }
            }
            
            // apply lr decay policy to compute eta
            const FloatType eta = this->lr_decay_->compute(iter); 
            // start inner loop
            for (std::size_t n = 0; n < this->num_inner_; ++n) {
                // check if we have to shuffle the samples
                if (this->shuffle_) {
                    sample_index = this->random_state_.sample<std::size_t>(X_data_index);
                }
                else {
                    sample_index = n;
                }
                
                // just-in-time update for full weight 1/n * sum(d(f_k_w))
                // n - update_history[j]: weight is not be updated for n - update_history[j] times
                // need to compensate the full weight update
                // update_history[j] = n: weight is already updated now
                // Update the weights to compensate for the full gradient
                if (n > 0) {
                    for (std::size_t j = 0; j < num_features; ++j) {
                        w0[j] += eta / wscale * (n - update_history[j]) * full_weight_update[j];
                        update_history[j] = n;
                    }
                }

                // compute current gradient at sample_index position
                // y_hat = w0 * X[sample_index : sample_index + num_features]
                y_hat = std::inner_product(&X[sample_index * num_features], 
                                           &X[(sample_index + 1) * num_features], 
                                           w0.begin(), 0.0);
                y_hat = y_hat * wscale;
                grad = this->loss_fn_->derivate(y_hat, y[sample_index]);

                // Update the weight scale factor based on the learning rate and regularization parameter
                wscale *= (1.0 - eta * this->alpha_);

                // update w0
                // grad_history[sample_index] - grad: reduce the variance of gradients
                for (std::size_t j = 0; j < num_features; ++j) {
                    w0[j] += eta * (grad_history[sample_index] - grad) / wscale;
                }

                // possible underflow
                // Check if the weight scale factor is below the threshold
                if (wscale < WSCALE_THRESHOLD) {
                    // Rescale the weights and reset the scale factor
                    std::transform(w0.begin(), w0.end(), w0.begin(),
                                   [wscale](FeatValType val) { 
                                       return val * wscale; 
                                   });
                    wscale = 1.0;
                }   
            }
            // Ensure that full gradient compensation is applied to all un-updated weights
            // Apply the full gradient compensation to any un-updated weights
            for (std::size_t j = 0; j < num_features; ++j) {
                w0[j] += eta / wscale * (this->num_inner_ - update_history[j]) * full_weight_update[j];
                update_history[j] = 0;
            }

            // check under/overflow
            if (sgdlib::internal::isinf<FeatValType>(w0)) {
                is_infinity = true;
                break;
            }

            // convergence test 1
            // Compute the L2 norm of the gradient
            fnorm = 0.0;
            for (std::size_t j = 0; j < num_features; ++j) {
                tmp = w0[j] * this->alpha_ + full_weight_update[j] / num_samples;
                fnorm += tmp * tmp;
            }
            fnorm = std::sqrt(fnorm);

            // Store the initial gradient norm
            if (iter == 0) {
                prev_fnorm = fnorm;
            }
            // Compute the ratio of the current gradient norm to the previous one
            fnorm_ratio = fnorm / prev_fnorm;
            
            if (fnorm_ratio < this->tol_) {
                is_converged = true;
                break;
            }
            // print info
            if (this->verbose_) {
                PRINT_RUNTIME_INFO(2, "Epoch = ", iter + 1, 
                                   ", fnorm_ration = ", fnorm_ratio);
            }
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

#endif // ALGORITHM_SGD_SVRG_HPP_