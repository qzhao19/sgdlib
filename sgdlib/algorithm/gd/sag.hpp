#ifndef ALGORITHM_GD_SAG_HPP_
#define ALGORITHM_GD_SAG_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

/**
 * @file sag.hpp
 *
 * @class SAG
 *
 * @brief Implements the Stochastic Average Gradient (SAG) optimization algorithm.
 *
 * This class inherits from `BaseOptimizer` and provides functionality for optimizing
 * machine learning models using the SAG algorithm. It also supports the SAGA variant,
 * which extends SAG to handle non-smooth regularization terms.
 *
*/
class SAG: public BaseOptimizer {
public:
    /**
     * @brief Constructor for the SAG optimizer.
     *
     * Initializes the SAG optimizer with the given parameters and passes them to the
     * base class `BaseOptimizer`.
     *
     * @param w0 Initial weight vector for the model.
     * @param b0 Initial bias term for the model.
     * @param loss The loss function to be minimized.
     * @param search_policy The policy for searching step size during optimization.
     * @param alpha L2 regularization parameter.
     * @param eta0 Initial learning rate.
     * @param tol Tolerance for convergence.
     * @param max_iters Maximum number of iterations for optimization.
     * @param random_seed Seed for the random number generator.
     * @param is_saga If true, enables the SAGA variant of the algorithm (default: false).
     * @param shuffle If true, shuffles the data before each epoch (default: true).
     * @param verbose If true, enables logging of optimization progress (default: true).
     *
     * @note This constructor calls the constructor of the base class `BaseOptimizer` to
     *       complete the initialization of the optimizer.
     * @see BaseOptimizer
    */
    SAG(const std::vector<FeatValType>& w0,
        const FeatValType& b0,
        std::string loss,
        std::string search_policy,
        FloatType alpha,
        FloatType eta0,
        FloatType tol,
        std::size_t max_iters,
        std::size_t random_seed,
        bool is_saga = false,
        bool shuffle = true,
        bool verbose = true): BaseOptimizer(w0, b0,
            loss,
            search_policy,
            alpha,
            eta0,
            tol,
            max_iters,
            random_seed,
            is_saga,
            shuffle,
            verbose) {};
    /**
     * Default destructor.
     */
    ~SAG() = default;

    void optimize(const std::vector<FeatValType>& X,
                  const std::vector<LabelValType>& y) override {

        const std::size_t num_samples = y.size();
        const std::size_t num_features = this->w0_.size();

        // initialize w0 (weight) and b0 (bias)
        std::vector<FeatValType> w0 = this->w0_;
        FeatValType b0 = this->b0_;

        // initialize gradient memory, the cumulative sums
        std::vector<FeatValType> grad_sum(num_features, 0.0);
        std::vector<FeatValType> grad_history(num_samples, 0.0);
        std::vector<FeatValType> cumulative_sum(this->max_iters_ * num_samples, 0.0);

        // array for visited samples
        std::vector<std::size_t> seen(num_samples, 0);
        std::vector<std::size_t> update_history(num_features, 0);

        std::size_t iter = 0;
        std::size_t num_seens = 0;
        std::size_t sample_index = 0;
        FeatValType inv_num_seens = 0.0;

        bool is_converged = false;
        bool is_infinity = false;
        int search_status = 0;
        FeatValType wscale = 1.0;

        // initialize a lookup table for training X, y
        this->X_data_index_.resize(num_samples);
        std::iota(this->X_data_index_.begin(), this->X_data_index_.end(), 0);

        // initialize loss, loss_history, gradient,
        FeatValType xnorm, wnorm;
        FeatValType y_hat, loss, dloss;
        FeatValType bias_update = 0.0;
        FeatValType grad_correction = 0.0;
        this->loss_history_.reserve(num_samples * this->max_iters_);
        std::vector<FeatValType> prev_weight(num_features, 0.0);
        std::vector<FeatValType> weight_update(num_features, 0.0);

        // compute step size
        FloatType step_size = 0.0;
        std::unique_ptr<sgdlib::detail::StepSizeSearch<sgdlib::detail::LossFunction>> stepsize_search;
        if (this->search_policy_ == "Constant") {
            stepsize_search = std::make_unique<sgdlib::detail::ConstantSearch<sgdlib::detail::LossFunction>>(
                X, y, this->loss_fn_, this->stepsize_search_params_
            );
            search_status = stepsize_search->search(is_saga_, step_size);
        }
        else if (this->search_policy_ == "BasicLineSearch") {
            stepsize_search = std::make_unique<sgdlib::detail::BasicLineSearch<sgdlib::detail::LossFunction>>(
                X, y, this->loss_fn_, this->stepsize_search_params_
            );
        }
        else {
            THROW_INVALID_ERROR("SAG optimizer supports 'Constant' or 'BasicLineSearch' policy only.");
        }

        std::size_t counter = 0;
        for (iter = 0; iter < this->max_iters_; ++iter) {
            // loop samples
            for (std::size_t i = 0; i < num_samples; ++i) {
                // check if we have to shuffle the samples
                if (this->shuffle_) {
                    sample_index = this->random_state_.sample<std::size_t>(this->X_data_index_);
                }
                else {
                    sample_index = i;
                }

                // update the number of X seen
                if (seen[sample_index] == 0) {
                    ++num_seens;
                    seen[sample_index] = 1;
                    inv_num_seens = 1.0 / static_cast<FeatValType>(num_seens);
                }

                // update weights
                if (counter >= 1) {
                    for (std::size_t j = 0; j < num_features; ++j) {
                        if (update_history[j] == 0) {
                            w0[j] -= cumulative_sum[counter - 1] * grad_sum[j];
                        }
                        else {
                            w0[j] -= (cumulative_sum[counter - 1] - cumulative_sum[update_history[j] - 1]) * grad_sum[j];
                        }
                        update_history[j] = counter;
                    }
                    if (sgdlib::detail::hasinf<FeatValType>(w0)) {
                        is_infinity = true;
                        break;
                    }
                }

                loss = 0.0;
                // compute loss value and its derivative (gradient) of this sample
                y_hat = sgdlib::detail::vecdot<FeatValType>(
                    X.data() + sample_index * num_features,
                    X.data() + (sample_index + 1) * num_features,
                    w0.data()
                );

                y_hat = y_hat * wscale + b0;
                loss  = this->loss_fn_->evaluate(y_hat, y[sample_index]);
                dloss = this->loss_fn_->derivate(y_hat, y[sample_index]);

                // stepsize-search step, apply basic line-search method
                // detail see section 4.6 of Schmidt, M., Roux, N., & Bach, F. (2013).
                // "Minimizing finite sums with the stochastic average gradient".
                if (search_policy_ == "BasicLineSearch") {
                    xnorm = sgdlib::detail::vecnorm2<FeatValType>(
                        X.data() + (sample_index * num_features),
                        X.data() + ((sample_index + 1) * num_features),
                        true
                    );
                    search_status = stepsize_search->search(y_hat, y[sample_index], dloss, xnorm, i, step_size);
                    if (search_status == -1) {
                        break;
                    }
                }

                // make the weight update to grad_sum
                // update = x * grad,
                sgdlib::detail::vecscale<FeatValType>(
                    X.data() + (sample_index * num_features),
                    X.data() + ((sample_index + 1) * num_features),
                    dloss,
                    weight_update
                );
                for (std::size_t j = 0; j < num_features; ++j) {
                    grad_correction = weight_update[j] - (grad_history[sample_index] * X[sample_index * num_features + j]);
                    grad_sum[j] += grad_correction;
                    if (this->is_saga_) {
                        w0[j] -= (grad_correction * step_size * (1.0 - inv_num_seens) / wscale);
                    }
                }

                // fit intercept
                grad_correction = dloss - grad_history[sample_index];
                bias_update += grad_correction;
                grad_correction *= step_size * (1.0 - inv_num_seens);
                if (this->is_saga_) {
                    b0 -= (step_size * bias_update * inv_num_seens) + grad_correction;
                }
                else {
                    b0 -= step_size * bias_update * inv_num_seens;
                }
                if (sgdlib::detail::isinf<FeatValType>(b0)) {
                    is_infinity = true;
                    break;
                }

                // update the gradient history for the current sample
                grad_history[sample_index] = dloss;

                if (counter == 0) {
                    cumulative_sum[0] = step_size / (wscale * num_seens);
                }
                else {
                    cumulative_sum[counter] = cumulative_sum[counter - 1] + step_size / (wscale * num_seens);
                }

                // if wscale is too small, need to reset
                if (counter >= 1 && wscale < WSCALE_THRESHOLD) {
                    for (std::size_t j = 0; j < num_features; ++j) {
                        if (update_history[j] == 0) {
                            w0[j] -= cumulative_sum[counter] * grad_sum[j];
                        }
                        else {
                            w0[j] -= (cumulative_sum[counter] - cumulative_sum[update_history[j] - 1]) * grad_sum[j];
                        }
                        update_history[j] = counter + 1;
                    }
                    cumulative_sum[counter] = 0.0;
                    sgdlib::detail::vecscale<FeatValType>(w0, wscale, w0);
                    wscale = 1.0;

                    if (sgdlib::detail::hasinf<FeatValType>(w0)) {
                        is_infinity = true;
                        break;
                    }
                }
                ++counter;

                // scale weight for L2 penalty
                if (this->alpha_ > 0.0) {
                    wscale *= 1.0 - this->alpha_ * step_size;
                    wnorm = sgdlib::detail::vecnorm2(w0, true);
                    loss += this->alpha_ * wnorm;
                }

                // store loss value into loss_history
                this->loss_history_.push_back(loss);
            }

            // break if raise an error in inner loop
            if (is_infinity || search_status == -1) {
                break;
            }

            // scale the weights
            for (std::size_t j = 0; j < num_features; ++j) {
                if (update_history[j] == 0) {
                    w0[j] -= cumulative_sum[counter - 1] * grad_sum[j];
                }
                else {
                    w0[j] -= (cumulative_sum[counter - 1] - cumulative_sum[update_history[j] - 1]) * grad_sum[j];
                }
            }
            sgdlib::detail::vecscale<FeatValType>(w0, wscale, w0);

            // compute loss info
            FeatValType sum_loss = sgdlib::detail::vecaccumul<FeatValType>(
                this->loss_history_.data() + (iter * num_samples),
                this->loss_history_.data() + ((iter + 1) * num_samples)
            );
            // check if convergence test is reached
            FeatValType max_change = 0.0, max_weight = 0.0;
            for (std::size_t j = 0; j < num_features; j++) {
                max_weight = std::max(max_weight, std::abs(w0[j]));
                max_change = std::max(max_change, std::abs(w0[j] - prev_weight[j]));
                prev_weight[j] = w0[j];
            }
            if ((max_weight != 0.0) && (max_change / max_weight <= tol_)) {
                if (this->verbose_) {
                    PRINT_RUNTIME_INFO(2, "Convergence after ", iter + 1, " epochs.");
                }
                is_converged = true;
                break;
            }
            else {
                if (this->verbose_) {
                    PRINT_RUNTIME_INFO(2, "Epoch = ", iter + 1,
                                       ", xnorm = ", sgdlib::detail::vecnorm2<FeatValType>(w0, true),
                                       ", loss = ", sum_loss / static_cast<FeatValType>(num_samples),
                                       ", change = ", max_change / max_weight);
                }
            }
        }
        // shrink the loss_history
        this->loss_history_.shrink_to_fit();

        // call callback function
        if (callback_) {
            callback_(this->loss_history_);
        }

        if (is_infinity) {
            THROW_RUNTIME_ERROR("Floating-point under-/overflow occurred at epoch ", (iter + 1),
                                ", try to scale input data with standard or minmax.");
        }

        if (!is_converged) {
            THROW_RUNTIME_ERROR("Not converge, current number of epoch ", (iter + 1),
                                ", try apply different parameters.");
        }

        if (search_status == -1) {
            THROW_RUNTIME_ERROR("Line-search condition not satisfied at epoch ", (iter + 1),
                                ", try apply different step-search parameters.");
        }

        this->w_opt_ = w0;
        this->b_opt_ = b0;
    }
};

} // namespace sgdlib

#endif // ALGORITHM_GD_SAG_HPP_
