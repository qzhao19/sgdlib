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

        // initialize loss, loss_history, gradient, 
        std::vector<FeatureType> grad(num_features);
        FeatureType loss, dloss;
        FeatureType y_hat;

        bool is_converged = false;
        FeatureType wscale = 1.0;

        // initialize a lookup table for training X, y
        std::vector<std::size_t> X_feature_index(num_samples);
        std::iota(X_feature_index.begin(), X_feature_index.end(), 0);

        std::size_t iter = 0;
        std::size_t f_index = 0;

        FeatureType eta = 0.0;
        std::size_t feat_index = 0;

        for (iter = 0; iter < max_iters_; ++iter) {
            // grad = loss_fn_->gradient(X_new, y_new, w0);

            for(std::size_t i = 0; i < num_samples; ++i) {
                y_hat = std::inner_product(&X[i * num_features], 
                                           &X[(i + 1) * num_features], 
                                           w0.begin(), 0.0);                    
                y_hat = y_hat * wscale + b0;
                dloss = loss_fn_->derivate(y_hat, y[i]);

                for(int j = 0; j < num_features; ++j) {
                    grad[j] += X[i * num_features + j] * dloss;
                }
            }
            for(int j = 0; j < num_features; ++j) {
                grad[j] /= num_samples;
            }

            FeatureType pred_descent = 0.0;
            FeatureType best_descent = -1.0;
            FeatureType best_eta = 0.0;
            std::size_t best_index = 0.0;
            
            for (feat_index = 0; feat_index < num_features; ++feat_index) {
                if ((w0[feat_index] - grad[feat_index] / l1_ratio_) > (alpha_ / l1_ratio_)) {
                    eta = (-grad[feat_index] / l1_ratio_) - (alpha_ / l1_ratio_);
                }
                else if ((w0(feat_index, 0) - grad[feat_index] / l1_ratio_) < (-alpha_ / l1_ratio_)) {
                    eta = (-grad[feat_index] / l1_ratio_) + (alpha_ / l1_ratio_);
                }
                else {
                    eta = -w0[feat_index];
                }

                pred_descent = -eta * grad[feat_index] - 
                    l1_ratio_ / 2 * eta * eta - 
                        alpha_ * std::abs(w0[feat_index] + eta) + 
                            alpha_ * std::abs(w0[feat_index]);

                if (pred_descent > best_descent) {
                    best_descent = pred_descent;
                    best_index = feat_index;
                    best_eta = eta;
                }

            }

            // update weight vector w
            w0[feat_index] += eta;

            std::fill_n(grad.begin(), num_features, 0); 
        }

    }

};

}

#endif // ALGORITHM_SGD_SCD_HPP_