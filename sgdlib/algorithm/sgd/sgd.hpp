#ifndef ALGORITHM_SGD_SGD_HPP_
#define ALGORITHM_SGD_SGD_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

/**
 * @file sgd.hpp
 * 
 * @brief Stochastic Gradient Descent (SGD) optimizer.
 *
*/
class SGD: public BaseOptimizer {
public:
    SGD(const std::vector<FeatValType>& w0, 
        const FeatValType& b0,
        std::string loss, 
        std::string lr_policy,
        FloatValType alpha,
        FloatValType eta0,
        FloatValType tol,
        FloatValType gamma,
        std::size_t max_iters, 
        std::size_t batch_size,
        std::size_t num_iters_no_change,
        std::size_t random_seed,
        bool shuffle = true, 
        bool verbose = true): BaseOptimizer(w0, b0,
            loss, lr_policy, 
            alpha, eta0, 
            tol, 
            gamma,
            max_iters, 
            batch_size, 
            num_iters_no_change,
            random_seed,
            shuffle, 
            verbose) {};
    ~SGD() {};

    void optimize(const std::vector<FeatValType>& X, 
                  const std::vector<LabelValType>& y) override {
        
        std::size_t num_samples = y.size();
        std::size_t num_features = this->w0_.size();
        std::size_t step_per_iter = num_samples / this->batch_size_;

        std::size_t no_improvement_count = 0;
        std::size_t iter = 0;

        bool is_converged = false;
        bool is_infinity = false;
        FloatValType best_loss = INF;
        FeatValType wscale = 1.0;

        // initialize a lookup table for training X, y
        std::vector<std::size_t> X_data_index(num_samples);
        std::iota(X_data_index.begin(), X_data_index.end(), 0);

        // initialize loss, loss_history, gradient, 
        FeatValType y_hat;
        FeatValType loss, dloss;
        FeatValType bias_update = 0.0;
        std::vector<FeatValType> loss_history(step_per_iter, 0.0);
        std::vector<FeatValType> weight_update(num_features, 0.0);
        
        // initialize w0 (weight) and b0 (bias)
        std::vector<FeatValType> w0 = this->w0_;
        FeatValType b0 = this->b0_;

        // start to loop
        for (iter = 0; iter < this->max_iters_; ++iter) {
            // enable to shuffle mask of data for on batch
            if (this->shuffle_) {
                this->random_state_.shuffle<std::size_t>(X_data_index);
            }

            // apply lr decay policy to compute eta
            FloatValType eta = this->lr_decay_->compute(iter);            
            for (std::size_t i = 0; i < step_per_iter; ++i) {
                for (std::size_t j = 0; j < this->batch_size_; ++j) {
                    // compute predicted label proba XW + b
                    y_hat = std::inner_product(&X[X_data_index[i * this->batch_size_ + j] * num_features], 
                                               &X[(X_data_index[i * this->batch_size_ + j] + 1) * num_features], 
                                               w0.begin(), 0.0);
                    y_hat = y_hat * wscale + b0;

                    // evaluate the loss on one row of X, and calculate the derivatives of the loss
                    loss += this->loss_fn_->evaluate(y_hat, y[X_data_index[i * this->batch_size_ + j]]);
                    dloss = this->loss_fn_->derivate(y_hat, y[X_data_index[i * this->batch_size_ + j]]);

                    // clip dloss with large values
                    sgdlib::internal::clip(dloss, -MAX_DLOSS, MAX_DLOSS);

                    if (dloss != 0.0) {
                        // Scales sample x by constant wscale and add it to weight:
                        // deflation of the sample feature values, adding to weights 
                        // means that this scaled sample directly affects the final output
                        dloss /= wscale;
                        sgdlib::internal::dot<FeatValType>(&X[X_data_index[i * this->batch_size_ + j] * num_features],
                                                           &X[(X_data_index[i * this->batch_size_ + j] + 1) * num_features],  
                                                           dloss, 
                                                           weight_update);
                        for (std::size_t k = 0; k < num_features; ++k) {
                            weight_update[k] += weight_update[k];
                        }
                        bias_update += dloss;
                    }

                    // scale weight vector by a scalar factor
                    wscale *= std::max(0.0, 1.0 - (eta * this->alpha_));
                    if (wscale < WSCALE_THRESHOLD) {
                        for (std::size_t k = 0; k < num_features; ++k) {
                            w0[k] *= wscale;
                        }
                        wscale = 1.0;
                    }
                }

                // compute loss/weight_gradient/bias_gradient for one batch data point
                if (this->batch_size_ > 1) {
                    loss /= static_cast<FeatValType>(this->batch_size_);
                    for (std::size_t k = 0; k < num_features; ++k) {
                        weight_update[k] /= static_cast<FeatValType>(this->batch_size_);
                    }
                    bias_update /= static_cast<FeatValType>(this->batch_size_);
                }
                
                // add L2 penalty for weight
                if (this->alpha_ > 0.0) {
                    loss += this->alpha_ * 
                        std::inner_product(w0.begin(), w0.end(), w0.begin(), 0.0) / 
                            static_cast<FeatValType>(num_samples);
                    for (std::size_t k = 0; k < num_features; ++k) {
                        weight_update[k] += (2.0 * this->alpha_ * w0[k] / static_cast<FeatValType>(num_samples));
                    }
                }

                // update w0: w = w - lr * w and b0: b = b - lr * b
                for (std::size_t k = 0; k < num_features; ++k) {
                    w0[k] -= eta * weight_update[k];
                }
                b0 -= eta * bias_update / (10.0 * static_cast<FeatValType>(num_samples));

                // store loss value into loss_history
                loss_history[i] = loss;
                loss = 0.0;
            }

            // ---Convergence test---
            // check under/overflow
            if (sgdlib::internal::isinf<FeatValType>(w0) || 
                sgdlib::internal::isinf<FeatValType>(b0)) {
                is_infinity = true;
                break;
            }

            // compute sum of the loss value for one full batch
            FeatValType sum_loss = std::accumulate(loss_history.begin(), 
                                                   loss_history.end(), 
                                                   decltype(loss_history)::value_type(0));
            if ((tol_ > -INF) && (sum_loss > best_loss - this->tol_ * num_samples)) {
                no_improvement_count +=1;
            }
            else {
                no_improvement_count = 0;
            }
            if (sum_loss < best_loss) {
                best_loss = sum_loss;
            }

            // if there is no improvement is bigger than the threshold
            if (no_improvement_count >= this->num_iters_no_change_) {
                if (this->verbose_) {
                    PRINT_RUNTIME_INFO(1, "Convergence after ", iter + 1, " epochs.");
                }
                is_converged = true;
                break;
            }

            if (this->verbose_) {
                PRINT_RUNTIME_INFO(2, "Epoch = ", iter + 1, 
                                   ", xnorm2 = ", sgdlib::internal::sqnorm2<FeatValType>(w0, true), 
                                   ", avg loss = ", sum_loss / static_cast<FeatValType>(step_per_iter));
            }
        }

        if (is_infinity) {
            THROW_RUNTIME_ERROR("Floating-point under-/overflow occurred at epoch ", (iter + 1),
                                ", try to scale input data with standard or minmax.");
        }

        if (!is_converged) {
            THROW_RUNTIME_ERROR("Not converge, current number of epoch ", (iter + 1),
                                ", the batch size ", this->batch_size_,
                                ", try to apply different parameters.");
        }

        this->w_opt_ = w0;
        this->b_opt_ = b0;
    }

};

} // namespace sgdlib

#endif // ALGORITHM_SGD_SGD_HPP_