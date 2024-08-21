#ifndef ALGORITHM_SGD_SCD_HPP_
#define ALGORITHM_SGD_SCD_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

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
        std::size_t num_features = w0.size();

        // initialize w0 (weight) and b0 (bias)
        std::vector<FeatureType> w0 = w0_;
        FeatureType b0 = b0_;

        // initialize loss, loss_history, weight_update, 
        // std::vector<FeatureType> weight_update(num_features);
        std::vector<FeatureType> xi_w(num_samples, 0.0);
        FeatureType weight_update, grad, dloss;
        // FeatureType y_hat;

        bool is_converged = false;
        FeatureType prev_weight;

        // initialize a lookup table for training X, y
        std::vector<std::size_t> X_feature_index(num_samples);
        std::iota(X_feature_index.begin(), X_feature_index.end(), 0);

        std::size_t iter = 0;
        // FeatureType eta = 0.0;
        std::size_t feature_index = 0;

        // compute column-wise norm2
        std::vector<FeatureType> X_col_norm(num_features);
        sgdlib::internal::col_norms<FeatureType>(X, true, X_col_norm);


        for (iter = 0; iter < max_iters_; ++iter) {
            // choose a feature index randomly
            feature_index = random_state_.random_index(0, num_features);

            // if norms of the columns of X is null
            if (X_col_norm[feature_index] == 0.0) {
                continue;
            }

            // record the previous weight
            prev_weight = w0[feature_index];

            dloss = 0.0;
            for (std::size_t i = 0; i < num_samples; ++i) {
                dloss += loss_fn_->derivate(xi_w[i], y[i]) * X[i * num_features + feature_index];
            }

            // compute gradient for target feature X[:, feature_index]
            grad = dloss / static_cast<FeatureType>(num_samples);

            // 
            if ((w0[feature_index] - grad / beta_) > (alpha_ / beta_)) {
                weight_update = -grad / beta_ - alpha_ / beta_;
            }
            else if ((w0[feature_index] - grad / beta_) < (-alpha_ / beta_)) {
                weight_update = -grad / beta_ + alpha_ / beta_;
            }
            else {
                weight_update = -w0[feature_index];
            }


            // update weight vector w
            w0[feature_index] += weight_update;

            // update inner product xi_w
            for (std::size_t i = 0; i < num_samples; ++i) {
                // dloss += loss_fn_->derivate(xi_w[i], y[i]) * X[i * feature_index];
                xi_w[i] = xi_w[i] + weight_update * X[i * num_features + feature_index];
            }



            // std::fill_n(weight_update.begin(), num_features, 0); 
        }

    }

};

}

#endif // ALGORITHM_SGD_SCD_HPP_