#ifndef ALGORITHM_CD_SCD_HPP_
#define ALGORITHM_CD_SCD_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

class SCD: public Optimizer {
public:
    SCD(const std::vector<sgdlib::FeatureScalarType>& w0,
        std::string loss,
        sgdlib::ScalarType alpha,
        sgdlib::ScalarType tol,
        std::size_t max_iters,
        std::size_t random_seed,
        bool shuffle = true,
        bool verbose = true): Optimizer(w0, loss, tol, max_iters, verbose) {
            this->alpha_ = alpha;
            this->random_seed_ = random_seed;
            this->shuffle_ = shuffle;
            this->init_random_state();
            this->init_loss_params();
            this->rho_ = (loss_ == "LogLoss") ? 0.25 : 1.0;
    }

    ~SCD() = default;

    void optimize(const sgdlib::ArrayDatasetType& dataset) override {
        const std::size_t num_samples = dataset.nrows();
        const std::size_t num_features = dataset.ncols();
        const sgdlib::FeatureScalarType inv_num_samples = 1.0 / static_cast<sgdlib::FeatureScalarType>(num_samples);
        // initialize w0 (weight)
        std::vector<sgdlib::FeatureScalarType> w0 = this->w0_;

        // initialize loss, loss_history, weight_update,
        this->loss_history_.reserve(this->max_iters_);
        std::vector<sgdlib::FeatureScalarType> y_hat(num_samples);
        sgdlib::FeatureScalarType weight_update, grad, loss, dloss;
        sgdlib::FeatureScalarType max_weight, max_weight_update;
        sgdlib::FeatureScalarType prev_weight;
        bool is_converged = false;

        // init x_j, y_i
        std::vector<sgdlib::LabelScalarType> y(num_samples);
        dataset.y_column_data(y);
        std::vector<sgdlib::FeatureScalarType> x_col(num_samples);
        std::size_t iter, feature_index;

        // compute column-wise norm2
        std::vector<sgdlib::FeatureScalarType> X_col_norm(num_features);
        sgdlib::detail::col_norms<sgdlib::FeatureScalarType>(dataset, false, X_col_norm);

        // start to loop
        for (iter = 0; iter < this->max_iters_; ++iter) {
            max_weight = 0.0;
            max_weight_update = 0.0;
            std::size_t best_feature_index;
            sgdlib::FeatureScalarType best_weight_update;
            sgdlib::FeatureScalarType best_descent = -1.0;
            sgdlib::FeatureScalarType pred_descent;

            // cycle through all the features
            for (int j = 0; j < num_features; ++j) {
                // choose a feature index randomly
                if(this->shuffle_) {
                    feature_index = this->random_state_.random_index(0, num_features);
                }
                else {
                    feature_index = j % num_features;
                }

                // if norms of the columns of X is null
                if (X_col_norm[feature_index] == 0.0) {
                    continue;
                }

                // compute gradient for target feature X[:, feature_index]
                dloss = 0.0;
                dataset.X_column_data(feature_index, x_col);
                #if defined(USE_OPENMP)
                #pragma omp parallel for reduction(+:dloss)
                #endif
                for (std::size_t i = 0; i < num_samples; ++i) {
                    dloss += this->loss_fn_->derivate(y_hat[i], y[i]) * x_col[i];
                }
                grad = dloss * inv_num_samples;

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

                // weight_update*grad: dot product of grad and weight_update
                // -this->rho_ / 2.0 * weight_update * weight_update: L2 regularization term
                //                                                      avoid too much w updates
                // -this->alpha_ * std::abs(w0[feature_index] + weight_update): L1 regularization term
                //                                                              for weights after updated
                // +this->alpha_ * std::abs(w0[feature_index]): L1 regularization term for current weight
                pred_descent = -weight_update * grad
                               - this->rho_ / 2.0 * weight_update * weight_update
                               - this->alpha_ * std::abs(w0[feature_index] + weight_update)
                               + this->alpha_ * std::abs(w0[feature_index]);

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

            // update inner product y_hat
            sgdlib::detail::vecadd<sgdlib::FeatureScalarType>(x_col, weight_update, y_hat);

            // compute loss
            loss = 0.0;
            for (std::size_t i = 0; i < num_samples; ++i) {
                loss += this->loss_fn_->evaluate(y_hat[i], y[i]);
            }
            loss *= inv_num_samples;

            // store loss value into loss_history
            this->loss_history_.push_back(loss);

            // print info
            if (this->verbose_) {
                PRINT_RUNTIME_INFO(5, "Epoch = ", iter + 1,
                                   ", wnorm1 = ", sgdlib::detail::vecnorm1<sgdlib::FeatureScalarType>(w0));
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
        // shrink the loss_history
        this->loss_history_.shrink_to_fit();

        // // call callback function
        if (callback_) {
            callback_(this->loss_history_);
        }

        if (!is_converged) {
            THROW_RUNTIME_ERROR("Not converge, current number of epoch ", (iter + 1),
                                ", try apply different parameters.");
        }
        this->w_opt_ = w0;
    }

    const sgdlib::FeatureScalarType get_intercept() const override {
        THROW_LOGIC_ERROR("The 'get_intercept' method is not supported for this SCD optimizer.");
        return 0.0;
    }
};

}
#endif // ALGORITHM_CD_SCD_HPP_
