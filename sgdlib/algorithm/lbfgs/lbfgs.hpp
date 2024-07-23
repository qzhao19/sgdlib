#ifndef ALGORITHM_LBFGS_LBFGS_HPP_
#define ALGORITHM_LBFGS_LBFGS_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

/**
 * @file lbfgs.hpp
 * 
 * @brief LBFGS optimizer.
 * 
*/
class LBFGS: public BaseOptimizer {
public:
    LBFGS(const std::vector<FeatureType>& w0, 
        const FeatureType& b0,
        std::string loss, 
        std::string linesearch_policy,
        double tol,
        std::size_t max_iters, 
        std::size_t mem_size,
        bool shuffle = true, 
        bool verbose = true): BaseOptimizer(w0, b0,
            loss, 
            linesearch_policy, 
            tol, 
            max_iters, 
            shuffle, 
            verbose) {};
    ~LBFGS() {};

    void optimize(const std::vector<FeatureType>& X, 
                  const std::vector<LabelType>& y) override {

        std::size_t num_samples = y.size();
        std::size_t num_features = w0_.size();

        // initialize w0 (weight) and b0 (bias)
        std::vector<FeatureType> w0 = w0_;
        FeatureType b0 = b0_;

        // 
    }
}

} // namespace sgdlib

#endif // ALGORITHM_LBFGS_LBFGS_HPP_