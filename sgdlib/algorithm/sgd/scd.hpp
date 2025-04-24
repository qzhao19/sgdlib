#ifndef ALGORITHM_SGD_SCD_HPP_
#define ALGORITHM_SGD_SCD_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {
/**
 * @file scd.hpp
 *
 * @class SCD
 *
 * @brief Implements the Stochastic Coordinate Descent (SCD) optimization algorithm.
 *
 * This class inherits from `BaseOptimizer` and provides functionality for optimizing
 * machine learning models using the Stochastic Coordinate Descent (SCD) algorithm.
 * SCD optimizes the model by updating one coordinate (feature weight) at a time,
 * making it efficient for high-dimensional problems.
 *
*/
class SCD: public BaseOptimizer {
public:
    /**
     * @brief Constructor for the SCD optimizer.
     *
     * Initializes the SCD optimizer with the given parameters and passes them to the
     * base class `BaseOptimizer`.
     *
     * @param w0 Initial weight vector for the model.
     * @param loss The loss function to be minimized.
     * @param alpha L1 regularization parameter.
     * @param tol Tolerance for convergence.
     * @param max_iters Maximum number of iterations for optimization.
     * @param random_seed Seed for the random number generator.
     * @param shuffle If true, shuffles the data before each epoch (default: true).
     * @param verbose If true, enables logging of optimization progress (default: true).
     *
     * @note This constructor calls the constructor of the base class `BaseOptimizer` to
     *       complete the initialization of the optimizer.
     * @see BaseOptimizer
     */
    SCD(const std::vector<FeatValType>& w0,
        std::string loss,
        FloatType alpha,
        FloatType tol,
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

    /**
     * @brief Destructor for the SCD optimizer.
     *
     * Default destructor.
    */
    ~SCD() = default;

    void optimize(const std::vector<FeatValType>& X,
                  const std::vector<LabelValType>& y) override {

        std::size_t num_samples = y.size();
        std::size_t num_features = this->w0_.size();

        // initialize w0 (weight)
        std::vector<FeatValType> w0 = this->w0_;

        // initialize loss, loss_history, weight_update,
        std::vector<FeatValType> loss_history;
        loss_history.reserve(this->max_iters_);
        std::vector<FeatValType> y_hat(num_samples, 0.0);
        FeatValType weight_update, grad, loss, dloss;

        // FeatValType y_hat;
        bool is_converged = false;
        FeatValType prev_weight;

        std::size_t iter = 0;
        std::size_t feature_index;

        // compute column-wise norm2
        std::vector<FeatValType> X_col_norm(num_features);
        sgdlib::detail::col_norms<FeatValType>(X, false, X_col_norm);

        FeatValType max_weight, max_weight_update;
        for (iter = 0; iter < this->max_iters_; ++iter) {

            max_weight = 0.0;
            max_weight_update = 0.0;
            std::size_t best_feature_index;
            FeatValType best_weight_update;
            FeatValType best_descent = -1.0;
            FeatValType pred_descent;

            // cycle through all the features
            for (feature_index = 0; feature_index < num_features; ++feature_index) {
                // choose a feature index randomly
                if(this->shuffle_) {
                    feature_index = this->random_state_.random_index(0, num_features);
                }

                // if norms of the columns of X is null
                if (X_col_norm[feature_index] == 0.0) {
                    continue;
                }

                // compute gradient for target feature X[:, feature_index]
                dloss = 0.0;
                for (std::size_t i = 0; i < num_samples; ++i) {
                    dloss += this->loss_fn_->derivate(y_hat[i], y[i]) * X[i * num_features + feature_index];
                }
                grad = dloss / static_cast<FeatValType>(num_samples);

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
                pred_descent = -weight_update*grad
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
            for (std::size_t i = 0; i < num_samples; ++i) {
                y_hat[i] += weight_update * X[i * num_features + feature_index];
            }

            // compute loss
            loss = 0.0;
            for (std::size_t i = 0; i < num_samples; ++i) {
                loss += this->loss_fn_->evaluate(y_hat[i], y[i]);
            }
            loss /= static_cast<FeatValType>(num_samples);
            // store loss value into loss_history
            loss_history.push_back(loss);

            // print info
            if (this->verbose_) {
                PRINT_RUNTIME_INFO(5, "Epoch = ", iter + 1,
                                   ", wnorm1 = ", sgdlib::detail::norm1<FeatValType>(w0) ,
                                   ", loss = ", loss);
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
        loss_history.shrink_to_fit();

        // call callback function
        if (callback_) {
            callback_(loss_history);
        }

        if (!is_converged) {
            THROW_RUNTIME_ERROR("Not converge, current number of epoch ", (iter + 1),
                                ", try apply different parameters.");
        }
        this->w_opt_ = w0;
    }

    const FeatValType get_intercept() const override {
        THROW_LOGIC_ERROR("The 'get_intercept' method is not supported for this SCD optimizer.");
        return 0.0;
    }
};

}
#endif // ALGORITHM_SGD_SCD_HPP_
