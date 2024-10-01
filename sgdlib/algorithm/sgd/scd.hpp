#ifndef ALGORITHM_SGD_SCD_HPP_
#define ALGORITHM_SGD_SCD_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {
/**
 * @file scd.hpp
 * 
 * @brief Stochastic Coordinate Descent (SCD) optimizer.
 * 
*/
class SCD: public BaseOptimizer {
public:
    SCD(const std::vector<FeatureType>& w0,
        const FeatureType& b0, 
        std::string loss,
        double alpha,
        double tol,
        std::size_t max_iters, 
        std::size_t random_seed,
        bool shuffle = true, 
        bool verbose = true): BaseOptimizer(w0, b0,
            loss, 
            alpha, 
            tol, 
            max_iters, 
            random_seed,
            shuffle, 
            verbose) {};

    ~SCD() {};

    void optimize(const std::vector<FeatureType>& X, 
                  const std::vector<LabelType>& y) override {
        
        std::size_t num_samples = y.size();
        std::size_t num_features = w0_.size();

        // initialize w0 (weight) and b0 (bias)
        std::vector<FeatureType> w0 = w0_;
        FeatureType b0 = b0_;

        // initialize loss, loss_history, weight_update, 
        // std::vector<FeatureType> weight_update(num_features);
        std::vector<FeatureType> xi_w(num_samples, 0.0);
        FeatureType weight_update, grad, loss, dloss;
        
        // FeatureType y_hat;
        bool is_converged = false;
        FeatureType prev_weight;

        std::size_t iter = 0;
        std::size_t feature_index = 0;

        // compute column-wise norm2
        std::vector<FeatureType> X_col_norm(num_features);
        sgdlib::internal::col_norms<FeatureType>(X, false, X_col_norm);


        FeatureType max_weight, max_weight_update;
         

        w_opt_ = w0;

    }
};

}
#endif // ALGORITHM_SGD_SCD_HPP_