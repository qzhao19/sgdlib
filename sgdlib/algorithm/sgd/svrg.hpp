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
            loss, lr_policy, 
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

        std::size_t num_samples = y.size();
        std::size_t num_features = this->w0_.size();
        std::size_t iter = 0, sample_index = 0;
        std::size_t batch_size = num_samples / this->num_inner_;

        // initialize a lookup table for training X, y
        std::vector<std::size_t> X_data_index(num_samples);
        std::iota(X_data_index.begin(), X_data_index.end(), 0);

        bool is_converged = false;
        bool is_infinity = false;
        FloatType best_loss = INF;
        FeatValType wscale = 1.0;
        // lr for regularization coefficient
        // FloatType eta_alpha = this->eta0_ * this->alpha_;

        // initialize a lookup table for training X, y
        std::vector<std::size_t> X_data_index(num_samples);
        std::iota(X_data_index.begin(), X_data_index.end(), 0);

        // initialize loss, loss_history, gradient, 
        FeatValType y_hat, dloss;
        std::vector<FeatValType> grad(num_samples, 0.0);
        // std::vector<FeatValType> loss_history(this->max_iters_, 0.0);
        std::vector<FeatValType> full_weight_update(num_features, 0.0);
        std::vector<std::size_t> update_history(num_features, 0);

        // initialize w0 (weight) and b0 (bias)
        std::vector<FeatValType> w0 = this->w0_;

        for (iter = 0; iter < this->max_iters_; ++iter) {
            // compute intial gradeint for full data
            // update full_weight_update is zero for beginning of iteration
            std::memset(full_weight_update.data(), 0, full_weight_update.size() * sizeof(FeatValType));
            for (std::size_t i = 0; i < num_samples; ++i) {
                y_hat = std::inner_product(&X[i * num_features], 
                                           &X[(i + 1) * num_features], 
                                           w0.begin(), 0.0);
                y_hat = y_hat * wscale;
                
                grad[i] = -this->loss_fn_->derivate(y_hat, y[i]);
                for (std::size_t j = 0; j < num_features; ++j) {
                    full_weight_update[j] += grad[i] * X[i * num_features + j];
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
                // n - update_history[j]: time differnece from last update, 
                // and accumulate the unapplied gradient
                if (n > 0) {
                    for (std::size_t j = 0; j < num_features; ++j) {
                        w0[j] += eta / wscale * (n - update_history[j]) * full_weight_update[j];
                        update_history[j] = n;
                    }
                }

                y_hat = std::inner_product(&X[sample_index * num_features], 
                                           &X[(sample_index + 1) * num_features], 
                                           w0.begin(), 0.0);
                y_hat = y_hat * wscale;

                dloss = -this->loss_fn_->derivate(y_hat, y[sample_index]);

                // wscale should be updated
                wscale *= (1.0 - eta);

                // update w0
                for (std::size_t j = 0; j < num_features; ++j) {
                    w0[j] += eta * (grad[sample_index] - dloss) / wscale;
                }

                // possible underflow
                if (wscale < WSCALE_THRESHOLD) {
                    std::transform(w0.begin(), w0.end(), w0.begin(),
                                   [wscale](FeatValType val) { 
                                       return val * wscale; 
                                   });
                    wscale = 1.0;
                }   
            }

            // convergence test


        }



    }
};

}

#endif // ALGORITHM_SGD_SVRG_HPP_