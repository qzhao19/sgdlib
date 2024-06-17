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
protected:
    void lagged_update(const std::vector<FeatureType>& sum_gradient,
                       const std::vector<std::size_t>& feature_hist,
                       const std::vector<std::size_t>& X_index_ptr, 
                       std::size_t num_samples, 
                       std::size_t num_features,
                       std::size_t sample_index,
                       double wscale,
                       bool reset,
                       std::vector<FeatureType>& x0,
                       std::vector<FeatureType>& cumulative_sums) {
        
    }

public:
    SAG(const std::vector<FeatureType>& x0, 
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
        bool verbose = true): BaseOptimizer(x0, 
            loss, lr_policy, 
            alpha, eta0, 
            tol, gamma,
            max_iters, 
            batch_size, 
            num_iters_no_change,
            random_seed,
            shuffle, 
            verbose) {};
    ~SAG() {};

    void optimize(const std::vector<FeatureType>& X, 
                  const std::vector<LabelType>& y) override {

        
    }


}

} // namespace sgdlib

#endif // ALGORITHM_SGD_SAG_HPP_