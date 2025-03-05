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
        std::string loss,
        double alpha,
        double tol,
        std::size_t max_iters, 
        std::size_t random_seed,
        bool shuffle = true, 
        bool verbose = true): BaseOptimizer(w0,
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
        std::size_t num_features = this->w0_.size();

        // initialize w0 (weight)
        std::vector<FeatureType> w0 = this->w0_;

        // initialize loss, loss_history, weight_update, 
        std::vector<FeatureType> xi_w(num_samples, 0.0);
        FeatureType weight_update, grad, loss, dloss;
        
        // FeatureType y_hat;
        bool is_converged = false;
        FeatureType prev_weight;

        std::size_t iter = 0;
        std::size_t feature_index;

        // compute column-wise norm2
        std::vector<FeatureType> X_col_norm(num_features);
        sgdlib::internal::col_norms<FeatureType>(X, false, X_col_norm);

        FeatureType max_weight, max_weight_update;
        for (iter = 0; iter < this->max_iters_; ++iter) {
            
            max_weight = 0.0;
            max_weight_update = 0.0;
            std::size_t best_feature_index;
            FeatureType best_weight_update;
            FeatureType best_descent = -1.0;
            FeatureType pred_descent;

            // cycle through all the features
            for (feature_index = 0; feature_index < num_features; ++feature_index) {
                // choose a feature index randomly
                feature_index = this->random_state_.random_index(0, num_features);
                
                // if norms of the columns of X is null
                if (X_col_norm[feature_index] == 0.0) {
                    continue;
                }

                // compute gradient for target feature X[:, feature_index]
                dloss = 0.0;
                for (std::size_t i = 0; i < num_samples; ++i) {
                    dloss += this->loss_fn_->derivate(xi_w[i], y[i]) * X[i * num_features + feature_index];
                }
                grad = dloss / static_cast<FeatureType>(num_samples);
                
                // soft-thresholding function
                if ((w0[feature_index] - grad / this->rho_) > (this->alpha_ / this->rho_)) {
                    weight_update = -grad / this->rho_ - this->alpha_ / this->rho_;
                }
                else if ((w0[feature_index] - grad / this->rho_) < (-this->alpha_ / this->rho_)) {
                    weight_update = -grad / this->rho_ + this->alpha_ / this->rho_;
                }
                else {
                    weight_update = -w0[feature_index];
                }
                
                pred_descent = -weight_update*grad - this->rho_ / 2.0 * weight_update * weight_update - \
                this->alpha_ * std::abs(w0[feature_index] + weight_update) + this->alpha_ * std::abs(w0[feature_index]);
                
                if (pred_descent > best_descent) {
                    best_feature_index = feature_index;
                    best_weight_update = weight_update;
                    best_descent = pred_descent;
                }
            }

            // update feature index and weight update
            feature_index = best_feature_index;
            weight_update = best_weight_update;

            // record the previous weight
            prev_weight = w0[feature_index];

            // update weight vector w
            w0[feature_index] = w0[feature_index] + weight_update;

            // max abs-coeff update
            max_weight = std::fmax(max_weight, w0[feature_index]); 
            max_weight_update = std::fmax(max_weight_update, std::abs(w0[feature_index] - prev_weight)); 

            // update inner product xi_w
            for (std::size_t i = 0; i < num_samples; ++i) {
                xi_w[i] += weight_update * X[i * num_features + feature_index];
            }

            // print info
            if (this->verbose_) {
                for (std::size_t i = 0; i < num_samples; ++i) {
                    loss += this->loss_fn_->evaluate(xi_w[i], y[i]);
                }
                PRINT_RUNTIME_INFO(5, "Epoch = ", iter + 1, 
                                   ", wnorm1 = ", sgdlib::internal::norm1<FeatureType>(w0) , 
                                   ", loss = ", loss / static_cast<FeatureType>(num_samples));
                loss = 0.0;
            }

            // convergence check maximum coordinate update
            // max_j|wj_new - wj_old| < tol * max_abs_coef_update
            // https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
            if ((max_weight_update / max_weight < this->tol_)) {
                is_converged = true;
                break;
            }

            if ((iter == this->max_iters_ - 1)) {
                is_converged = false;
            }
        }
        if (!is_converged) {
            THROW_RUNTIME_ERROR("Not converge, current number of epoch ", (iter + 1), 
                                ", try apply different parameters.");
        }
        this->w_opt_ = w0;

    }

    const FeatureType get_intercept() const override {
        THROW_RUNTIME_ERROR("Not support to call get_intercept method.");
        return 0.0;
    }
};

}
#endif // ALGORITHM_SGD_SCD_HPP_