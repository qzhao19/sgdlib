#ifndef ALGORITHM_SGD_SGD_HPP_
#define ALGORITHM_SGD_SGD_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

/**
 * @file sgd.hpp
 * 
 * @brief Stochastic Gradient Descent (SGD) optimizer.
 * 
 * @param w0 FeatureType vector. 
 *      The initial weight vector of model.
 * @param b0 FeatureType scalar. 
 *      The initial bias scalar of model.
 * @param loss String. 
 *      The type of loss function to be used for optimization, e.g., "log_loss".
 * @param lr_policy String
 *      The learning rate schedule policy, which can be "exponential" or "invscaling".
 * @param alpha Double
 *      The regularization strength, which penalizes large parameter values to prevent overfitting.
 *      The higher the alpha, the stronger the regularization.
 * @param eta0 Double. 
 *      The initial value of the learning rate.
 * @param tol Double 
 *      The convergence tolerance, which is a threshold below which the change in the loss function
 *      indicates that the optimizer has likely converged.
 * @param gamma Double 
 *      A hyperparameter that controls the decay of learning rate.
 * @param max_iters Integer. 
 *      The maximum number of iterations (epochs) that the optimizer will run before stopping.
 * @param batch_size Integer. 
 *      The number of samples to be processed in each iteration of the optimization loop.
 *      Smaller batch sizes can provide a regularizing effect, while larger batch sizes may 
 *      offer computational efficiency.
 * @param num_iters_no_change Integer. 
 *      The number of iterations after which the optimizer will stop if there is no
 *      improvement in the loss function.
 * @param random_seed Integer 
 *      The seed for the random number generator, which ensures reproducibility of the results
 *      when shuffling the data or selecting batches.
 * @param shuffle Boolean. 
 *      Whether to randomly shuffle the data before starting each iteration. This can help prevent
 *      the optimizer from converging to a local minimum and is generally recommended.
 * @param verbose Boolean 
 *      Whether to output detailed information during the optimization process, such as the progress,
 *      loss values, and parameter updates.
 *
*/
class SGD: public BaseOptimizer {
public:
    SGD(const std::vector<FeatureType>& w0, 
        const FeatureType& b0,
        std::string loss, 
        std::string lr_policy,
        double alpha,
        double eta0,
        double tol,
        double gamma,
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

    void optimize(const std::vector<FeatureType>& X, 
                  const std::vector<LabelType>& y) override {
        
        std::size_t num_samples = y.size();
        std::size_t num_features = w0_.size();
        std::size_t step_per_iter = num_samples / batch_size_;

        std::size_t no_improvement_count = 0;
        std::size_t iter = 0;

        bool is_converged = false;
        bool is_infinity = false;
        double best_loss = INF;
        FeatureType wscale = 1.0;

        // initialize a lookup table for training X, y
        std::vector<std::size_t> X_data_index(num_samples);
        std::iota(X_data_index.begin(), X_data_index.end(), 0);

        // initialize loss, loss_history, gradient, 
        FeatureType y_hat;
        FeatureType loss, dloss;
        FeatureType bias_update = 0.0;
        std::vector<FeatureType> loss_history(step_per_iter, 0.0);
        std::vector<FeatureType> weight_update(num_features, 0.0);
        
        // initialize w0 (weight) and b0 (bias)
        std::vector<FeatureType> w0 = w0_;
        FeatureType b0 = b0_;

        // start to loop
        for (iter = 0; iter < max_iters_; ++iter) {
            // enable to shuffle mask of data for on batch
            if (shuffle_) {
                random_state_.shuffle<std::size_t>(X_data_index);
            }

            // apply lr decay policy to compute eta
            double eta = lr_decay_->compute(iter);            
            for (std::size_t i = 0; i < step_per_iter; ++i) {
                for (std::size_t j = 0; j < batch_size_; ++j) {
                    // compute predicted label proba XW + b
                    y_hat = std::inner_product(&X[X_data_index[i * batch_size_ + j] * num_features], 
                                               &X[(X_data_index[i * batch_size_ + j] + 1) * num_features], 
                                               w0.begin(), 0.0);                    
                    y_hat = y_hat * wscale + b0;

                    // evaluate the loss on one row of X, and calculate the derivatives of the loss
                    loss += loss_fn_->evaluate(y_hat, y[X_data_index[i * batch_size_ + j]]);
                    dloss = loss_fn_->derivate(y_hat, y[X_data_index[i * batch_size_ + j]]);

                    // clip dloss with large values
                    sgdlib::internal::clip(dloss, -MAX_DLOSS, MAX_DLOSS);

                    if (dloss != 0.0) {
                        // Scales sample x by constant wscale and add it to weight:
                        // deflation of the sample feature values, adding to weights 
                        // means that this scaled sample directly affects the final output
                        dloss /= wscale;
                        sgdlib::internal::dot<FeatureType>(&X[X_data_index[i * batch_size_ + j] * num_features],
                                                           &X[(X_data_index[i * batch_size_ + j] + 1) * num_features],  
                                                           dloss, 
                                                           weight_update);
                        for (std::size_t k = 0; k < num_features; ++k) {
                            weight_update[k] += weight_update[k];
                        }
                        bias_update += dloss;
                    }

                    // scale weight vector by a scalar factor
                    wscale *= std::max(0.0, 1.0 - (eta * alpha_));
                    if (wscale < WSCALE_THRESHOLD) {
                        for (std::size_t k = 0; k < num_features; ++k) {
                            w0[k] *= wscale;
                        }
                        wscale = 1.0;
                    }
                }

                // compute loss/weight_gradient/bias_gradient for one batch data point
                if (batch_size_ > 1) {
                    loss /= static_cast<FeatureType>(batch_size_);
                    for (std::size_t k = 0; k < num_features; ++k) {
                        weight_update[k] /= static_cast<FeatureType>(batch_size_);
                    }
                    bias_update /= static_cast<FeatureType>(batch_size_);
                }
                
                // add L2 penalty for weight
                if (alpha_ > 0.0) {
                    loss += alpha_ * 
                        std::inner_product(w0.begin(), w0.end(), w0.begin(), 0.0) / 
                            static_cast<FeatureType>(num_samples);
                    for (std::size_t k = 0; k < num_features; ++k) {
                        weight_update[k] += (2.0 * alpha_ * w0[k] / static_cast<FeatureType>(num_samples));
                    }
                }

                // update w0: w = w - lr * w and b0: b = b - lr * b
                for (std::size_t k = 0; k < num_features; ++k) {
                    w0[k] -= eta * weight_update[k];
                }
                b0 -= eta * bias_update / (10.0 * static_cast<FeatureType>(num_samples));

                // store loss value into loss_history
                loss_history[i] = loss;
                loss = 0.0;
            }

            // ---Convergence test---
            // check under/overflow
            if (sgdlib::internal::isinf<FeatureType>(w0) || 
                sgdlib::internal::isinf<FeatureType>(b0)) {
                is_infinity = true;
                break;
            }

            // compute sum of the loss value for one full batch
            FeatureType sum_loss = std::accumulate(loss_history.begin(), 
                                                   loss_history.end(), 
                                                   decltype(loss_history)::value_type(0));
            if ((tol_ > -INF) && (sum_loss > best_loss - tol_ * num_samples)) {
                no_improvement_count +=1;
            }
            else {
                no_improvement_count = 0;
            }
            if (sum_loss < best_loss) {
                best_loss = sum_loss;
            }

            // if there is no improvement is bigger than the threshold
            if (no_improvement_count >= num_iters_no_change_) {
                if (verbose_) {
                    std::cout << "Convergence after " << (iter + 1) << " epochs." << std::endl;
                }
                is_converged = true;
                break;
            }

            if (verbose_) {
                if ((iter % 1) == 0) {
                    std::cout << "Epoch = " << (iter + 1) << ", xnorm2 = " 
                              << sgdlib::internal::sqnorm2<FeatureType>(w0) << ", avg loss = " 
                              << sum_loss / static_cast<FeatureType>(step_per_iter) << std::endl;
                }
            }
        }

        if (is_infinity) {
            std::ostringstream err_msg;
            err_msg << "Floating-point under-/overflow occurred at epoch " << (iter + 1)
                    << ", try to scale input data with standard or minmax." << std::endl;
            throw std::runtime_error(err_msg.str());
        }

        if (!is_converged) {
            std::ostringstream err_msg;
            err_msg << "Not converge, current number of epoch = " << (iter + 1)
                    << ", the batch size = " << batch_size_ 
                    << ", try apply different parameters." << std::endl;
            throw std::runtime_error(err_msg.str());
        }

        w_opt_ = w0;
        b_opt_ = b0;
    }

};

} // namespace sgdlib

#endif // ALGORITHM_SGD_SGD_HPP_