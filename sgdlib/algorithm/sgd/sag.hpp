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

        std::size_t num_samples = y.size();
        std::size_t num_features = x0_.size();

        std::vector<FeatureType> x0 = x0_;
        std::vector<FeatureType> sum_grad(num_features);
        std::vector<FeatureType> grad_history(num_features);
        std::vector<FeatureType> last_updated(num_features);

    }


}

} // namespace sgdlib

#endif // ALGORITHM_SGD_SAG_HPP_