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
        
        for (iter = 0; iter < max_iters_; ++iter) {
            max_weight = 0.0;
            max_weight_update = 0.0;
            std::size_t best_feature_index;
            FeatureType best_weight_update;
            FeatureType best_descent = -1.0;
            FeatureType pred_descent;
            // cycle through all the features
            for (std::size_t j = 0; j < num_features; ++j) {
                // choose a feature index randomly
                // feature_index = random_state_.random_index(0, num_features);
                feature_index = j;
                // if norms of the columns of X is null
                if (X_col_norm[feature_index] == 0.0) {
                    continue;
                }
                // record the previous weight
                prev_weight = w0[feature_index];
                dloss = 0.0;
                for (std::size_t i = 0; i < num_samples; ++i) {
                    // std::cout << "xi_w[i] = " << xi_w[i] << std::endl;
                    // std::cout << "loss_fn_->derivate(xi_w[i], y[i]) = " << loss_fn_->derivate(0.0, 1) << std::endl;
                    dloss += loss_fn_->derivate(xi_w[i], y[i]) * X[i * num_features + feature_index];
                }
                // std::cout << "dloss = " << dloss << std::endl;
                // compute gradient for target feature X[:, feature_index]
                grad = dloss / static_cast<FeatureType>(num_samples);
                // soft-thresholding function
                if ((w0[feature_index] - grad / rho_) > (alpha_ / rho_)) {
                    weight_update = -grad / rho_ - alpha_ / rho_;
                }
                else if ((w0[feature_index] - grad / rho_) < (-alpha_ / rho_)) {
                    weight_update = -grad / rho_ + alpha_ / rho_;
                }
                else {
                    weight_update = -w0[feature_index];
                }
                pred_descent = -weight_update*grad - rho_ / 2.0 * weight_update * weight_update - \
                    alpha_ * (std::abs(w0[feature_index] + weight_update) - std::abs(w0[feature_index]));
                if (pred_descent > best_descent) {
                    best_feature_index = feature_index;
                    best_weight_update = weight_update;
                    best_descent = pred_descent;
                }
                // max abs-coeff update
                max_weight = std::fmax(max_weight, w0[feature_index]); 
                max_weight_update = std::fmax(max_weight_update, std::abs(w0[feature_index] - prev_weight)); 
            }
            feature_index = best_feature_index;
            weight_update = best_weight_update;
            // update weight vector w
            w0[feature_index] += weight_update;
            // update inner product xi_w
            for (std::size_t i = 0; i < num_samples; ++i) {
                xi_w[i] = xi_w[i] + weight_update * X[i * num_features + feature_index];
            }
            // print info
            if (verbose_) {
                for (std::size_t i = 0; i < num_samples; ++i) {
                    loss += loss_fn_->evaluate(xi_w[i], y[i]);
                }
                std::cout << "Epoch = " << (iter + 1) << ", wnorm1 = " 
                          << sgdlib::internal::norm1<FeatureType>(w0) << ", avg loss = " 
                          << loss / static_cast<FeatureType>(num_samples) << std::endl;
            }
            // convergence check maximum coordinate update
            // max_j|wj_new - wj_old| < tol * max_abs_coef_update
            // https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
            if ((max_weight_update / max_weight > tol_) || (iter == max_iters_ - 1)) {
                is_converged = false;
                break;
            }
        }
        if (!is_converged) {
            std::ostringstream err_msg;
            err_msg << "Not converge, current number of epoch = " << (iter + 1)
                    << ", try apply different parameters." << std::endl;
            throw std::runtime_error(err_msg.str());
        }

        w_opt_ = w0;

    }
};

}
#endif // ALGORITHM_SGD_SCD_HPP_