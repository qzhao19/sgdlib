#ifndef ALGORITHM_SGD_SAG_HPP_
#define ALGORITHM_SGD_SAG_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

/**
 * @file sag.hpp
 * 
 * @brief Stochastic Average Gradient Descent (SAGD) optimizer.
 * 
*/
class SAG: public BaseOptimizer {
public:
    SAG(const std::vector<FeatureType>& w0, 
        const FeatureType& b0,
        std::string loss, 
        double alpha,
        double tol,
        std::size_t max_iters, 
        std::size_t random_seed,
        bool is_saga = false,
        bool shuffle = true, 
        bool verbose = true): BaseOptimizer(w0, b0,
            loss, 
            alpha, 
            tol, 
            max_iters, 
            random_seed,
            is_saga,
            shuffle, 
            verbose) {};
    ~SAG() {};

    void optimize(const std::vector<FeatureType>& X, 
                  const std::vector<LabelType>& y) override {

        std::size_t num_samples = y.size();
        std::size_t num_features = w0_.size();

        // initialize w0 (weight) and b0 (bias)
        std::vector<FeatureType> w0 = w0_;
        FeatureType b0 = b0_;

        // 
        FeatureType bias_sum = 0.0;
        std::vector<FeatureType> grad_sum(num_features, 0.0);
        std::vector<FeatureType> grad_history(num_samples, 0.0);
        std::vector<FeatureType> cumulative_sum(max_iters_ * num_samples, 0.0);

        std::vector<std::size_t> seen(num_samples, 0);
        std::vector<std::size_t> update_history(num_features, 0);
        
        std::size_t iter = 0;
        std::size_t num_seens = 0;
        std::size_t sample_index = 0;

        bool is_converged = false;
        bool is_infinity = false;
        FeatureType wscale = 1.0;

        // initialize a lookup table for training X, y
        std::vector<std::size_t> X_data_index(num_samples);
        std::iota(X_data_index.begin(), X_data_index.end(), 0);

        // initialize loss, loss_history, gradient, 
        FeatureType loss, dloss;
        FeatureType y_hat;
        FeatureType bias_update = 0.0;
        FeatureType grad_correction = 0.0;
        std::vector<FeatureType> loss_history(num_samples, 0.0);
        std::vector<FeatureType> prev_weight(num_features, 0.0);
        std::vector<FeatureType> weight_update(num_features, 0.0);
        
        // compute step size 
        double step_size = 0.0;
        std::unique_ptr<sgdlib::StepSizeSearch> stepsize_search_ = \
            std::make_unique<sgdlib::ConstantSearch>(X, y, stepsize_search_param_);
        stepsize_search_->search(is_saga_, step_size);

        std::size_t counter = 0;
        for (iter = 0; iter < max_iters_; ++iter) {
            for (std::size_t i = 0; i < num_samples; ++i) {
                sample_index = random_state_.sample<std::size_t>(X_data_index);
                
                // update the number of X seen
                if (seen[sample_index] == 0) {
                    ++num_seens;
                    seen[sample_index] = 1;
                }

                // update weights
                if (counter >= 1) {
                    for (std::size_t j = 0; j < num_features; ++j) {
                        if (update_history[j] == 0) {
                            w0[j] -= cumulative_sum[counter - 1] * grad_sum[j];
                        }
                        else {
                            w0[j] -= (cumulative_sum[counter - 1] - cumulative_sum[update_history[j] - 1]) * grad_sum[j];
                        }
                        update_history[j] = counter;
                    }
                    if (sgdlib::internal::isinf<FeatureType>(w0)) {
                        is_infinity = true;
                        break;
                    }
                }

                // compute loss value and its derivative (gradient) of this sample
                y_hat = std::inner_product(&X[sample_index * num_features], 
                                           &X[(sample_index + 1) * num_features], 
                                           w0.begin(), 0.0);                    
                y_hat = y_hat * wscale + b0;
                loss  = loss_fn_->evaluate(y_hat, y[sample_index]);
                dloss = loss_fn_->derivate(y_hat, y[sample_index]);

                // make the weight update to grad_sum
                // update = x * grad, 
                sgdlib::internal::dot<FeatureType>(&X[sample_index * num_features], 
                                                   &X[(sample_index + 1) * num_features], 
                                                   dloss,
                                                   weight_update);
                for (std::size_t j = 0; j < num_features; ++j) {
                    grad_correction = weight_update[j] - (grad_history[sample_index] * X[sample_index * num_features + j]);
                    grad_sum[j] += grad_correction;
                    if (is_saga_) {
                        w0[j] -= (grad_correction * step_size * (1.0 - 1.0 / static_cast<FeatureType>(num_seens)) / wscale);
                    }
                }

                // fit intercept
                grad_correction = dloss - grad_history[sample_index];
                bias_sum += grad_correction;
                grad_correction *= step_size * (1.0 - 1.0 / static_cast<FeatureType>(num_seens));
                if (is_saga_) {
                    b0 -= (step_size * bias_sum / static_cast<FeatureType>(num_seens)) + grad_correction;
                }
                else {
                    b0 -= step_size * bias_sum / static_cast<FeatureType>(num_seens);
                }
                if (sgdlib::internal::isinf<FeatureType>(b0)) {
                    is_infinity = true;
                    break;
                }

                // update the gradient history for the current sample
                grad_history[sample_index] = dloss;

                if (counter == 0) {
                    cumulative_sum[0] = step_size / (wscale * num_seens);
                }
                else {
                    cumulative_sum[counter] = cumulative_sum[counter - 1] + step_size / (wscale * num_seens); 
                }

                // if wscale is too small, need to reset 
                if (counter >= 1 && wscale < WSCALE_THRESHOLD) {
                    for (std::size_t j = 0; j < num_features; ++j) {
                        if (update_history[j] == 0) {
                            w0[j] -= cumulative_sum[counter] * grad_sum[j];
                        }
                        else {
                            w0[j] -= (cumulative_sum[counter] - cumulative_sum[update_history[j] - 1]) * grad_sum[j];
                        }
                        update_history[j] = counter + 1;
                    }
                    cumulative_sum[counter] = 0.0;
                    sgdlib::internal::dot<FeatureType>(w0, wscale);
                    wscale = 1.0;

                    if (sgdlib::internal::isinf<FeatureType>(w0)) {
                        is_infinity = true;
                        break;
                    }
                }
                ++counter;
                
                // scale weight for L2 penalty
                if (alpha_ > 0.0) {
                    wscale *= 1.0 - alpha_ * step_size;
                    loss += alpha_ * std::inner_product(w0.begin(), w0.end(), w0.begin(), 0.0);
                }

                // store loss value into loss_history
                loss_history[i] = loss;
                loss = 0.0;
            }

            // break if raise an error in inner loop
            if (is_infinity) {
                break;
            }

            // scale the weights
            for (std::size_t j = 0; j < num_features; ++j) {
                if (update_history[j] == 0) {
                    w0[j] -= cumulative_sum[counter - 1] * grad_sum[j];
                }
                else {
                    w0[j] -= (cumulative_sum[counter - 1] - cumulative_sum[update_history[j] - 1]) * grad_sum[j];
                }
            }
            sgdlib::internal::dot<FeatureType>(w0, wscale);

            // calc loss info
            FeatureType sum_loss = std::accumulate(loss_history.begin(), 
                                                   loss_history.end(), 
                                                   decltype(loss_history)::value_type(0));
            // check if convergence test is reached
            FeatureType max_change = 0.0, max_weight = 0.0;
            for (std::size_t j = 0; j < num_features; j++) {
                max_weight = std::max(max_weight, std::abs(w0[j]));
                max_change = std::max(max_change, std::abs(w0[j] - prev_weight[j]));
                prev_weight[j] = w0[j];
            }
            if ((max_weight != 0.0) && (max_change / max_weight <= tol_)) {
                if (verbose_) {
                    std::cout << "Convergence after " << (iter + 1) << " epochs." << std::endl;
                }
                is_converged = true;
                break;
            }
            else {
                if (verbose_) {
                    std::cout << "Epoch = " << iter 
                              << ", xnorm = " << sgdlib::internal::sqnorm2<FeatureType>(w0) 
                              << ", loss = " << sum_loss / static_cast<FeatureType>(num_samples) 
                              << ", change = " << max_change / max_weight << std::endl;
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
                    << ", try apply different parameters." << std::endl;
            throw std::runtime_error(err_msg.str());
        }

        w_opt_ = w0;
        b_opt_ = b0;
    }
};

} // namespace sgdlib

#endif // ALGORITHM_SGD_SAG_HPP_