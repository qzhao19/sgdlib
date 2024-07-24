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
        std::string loss, 
        std::string linesearch_policy,
        double tol,
        std::size_t max_iters, 
        std::size_t mem_size,
        std::size_t past,
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
        std::size_t num_dims = w0_.size();

        // initialize w0 (weight)
        std::vector<FeatureType> w0 = w0_;

        // define the initial parameters
        std::size_t i, j, k, end, bound;
        double fx, ys, yy, rate, beta;

        // intermediate variables: previous x, gradient, previous gradient, directions
        std::vector<FeatureType> xp(num_dims);
        std::vector<FeatureType> g(num_dims);
        std::vector<FeatureType> gp(num_dims);
        std::vector<FeatureType> d(num_dims);

        // an array for storing previous values of the objective function
        std::vector<FeatureType> pfx(std::max(1, past_));

        // define step search policy
    }
}

} // namespace sgdlib

#endif // ALGORITHM_LBFGS_LBFGS_HPP_