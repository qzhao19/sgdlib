#ifndef ALGORITHM_SGD_SAG_HPP_
#define ALGORITHM_SGD_SAG_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

/**
 * @file sag.hpp
 * 
 * @brief @brief Stochastic Average Gradient Descent (SAGD) optimizer.
 * 
 * 
*/
class SAG: public BaseOptimizer {
public:
    SAG(const std::vector<FeatureType>& x0, 
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
        bool verbose = true): BaseOptimizer(x0, b0,
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
    ~SAG() {};

    void optimize(const std::vector<FeatureType>& X, 
                  const std::vector<LabelType>& y) override {

        std::size_t num_samples = y.size();
        std::size_t num_features = x0_.size();

        // initialize x0 (weight) and b0 (bias)
        std::vector<FeatureType> x0 = x0_;
        FeatureType b0 = b0_;

        // 
        std::vector<FeatureType> grad_sum(num_features);
        std::vector<FeatureType> grad_history(num_features);
        std::vector<FeatureType> cumulative_sum(max_iters_ * num_samples);

        std::vector<std::size_t> seen(num_samples, 0);
        std::vector<std::size_t> last_updated(num_features, 0);
        
        std::size_t no_improvement_count = 0;
        std::size_t sample_index = 0;
        std::size_t iter = 0;
        std::size_t num_seens = 0;

        bool is_converged = false;
        bool is_infinity = false;
        double best_loss = INF;
        FeatureType wscale = 1.0;

        // initialize a lookup table for training X, y
        std::vector<std::size_t> X_data_index(num_samples);
        std::iota(X_data_index.begin(), X_data_index.end(), 0);

        // initialize loss, loss_history, gradient, 
        double loss, dloss;
        FeatureType y_hat;
        FeatureType bias_update = 0.0;
        std::vector<double> loss_history(num_samples, 0.0);
        std::vector<FeatureType> weight_update(num_features, 0.0);

        std::size_t counter = 0;
        for (iter = 0; iter < max_iters_; ++iter) {
            for (std::size_t i = 0; i < num_samples; ++i) {
                sample_index = random_state_.random<std::size_t>(X_data_index);

                // update the number of X seen
                if (seen[sample_index] == 0) {
                    ++num_seens;
                    seen[sample_index] = 1;
                }
                
                if (counter >= 1) {
                    for (std::size_t j = 0; j < num_features; ++j) {
                        if (last_updated[j] == 0) {
                            x0[j] -= cumulative_sum[j-1] * grad_sum[j];
                        }
                        else {
                            x0[j] -= (cumulative_sum[j-1] - cumulative_sum[last_updated[j] - 1]) * grad_sum[j];
                        }
                        last_updated[j] = counter;
                    }
                }

                y_hat = std::inner_product(&X[sample_index * num_features], 
                                           &X[(sample_index + 1) * num_features], 
                                           x0.begin(), 0.0);                    
                y_hat = y_hat * wscale + b0;

                dloss = loss_fn_->derivate(y_hat, y[sample_index]);

                std::transform(&X[sample_index * num_features], 
                               &X[(sample_index + 1) * num_features], 
                               weight_update.begin(),
                               [dloss](FeatureType elem) {return elem * dloss;});

            }


        }

    }


}

} // namespace sgdlib

#endif // ALGORITHM_SGD_SAG_HPP_