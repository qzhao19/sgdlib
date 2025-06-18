#ifndef ALGORITHM_GD_SGD_HPP_
#define ALGORITHM_GD_SGD_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

class SGD: public Optimizer {
public:
    SGD(const std::vector<sgdlib::FeatureScalarType>& w0,
        const sgdlib::FeatureScalarType& b0,
        std::string loss,
        std::string lr_policy,
        sgdlib::ScalarType alpha,
        sgdlib::ScalarType eta0,
        sgdlib::ScalarType tol,
        sgdlib::ScalarType gamma,
        std::size_t max_iters,
        std::size_t batch_size,
        std::size_t num_iters_no_change,
        std::size_t random_seed,
        bool shuffle = true,
        bool verbose = true): Optimizer(w0, loss, tol, max_iters, verbose) {
            this->b0_ = b0;
            this->lr_policy_ = lr_policy;
            this->alpha_ = alpha;
            this->eta0_ = eta0;
            this->gamma_ = gamma;
            this->batch_size_ = batch_size;
            this->num_iters_no_change_ = num_iters_no_change;
            this->random_seed_ = random_seed;
            this->shuffle_ = shuffle;
            this->init_random_state();
            this->init_lr_params();
        };

    ~SGD() = default;

    void optimize(const sgdlib::ArrayDatasetType& dataset) override {
        const std::size_t num_samples = dataset.nrows();
        const std::size_t num_features = dataset.ncols();

        const std::size_t step_per_iter = num_samples / this->batch_size_;
        const sgdlib::FeatureScalarType inv_num_samples = 1.0 / static_cast<sgdlib::FeatureScalarType>(num_samples);
        const sgdlib::FeatureScalarType inv_batch_size = 1.0 / static_cast<sgdlib::FeatureScalarType>(this->batch_size_);

        std::size_t no_improvement_count = 0;
        std::size_t iter = 0;

        bool is_converged = false;
        bool is_infinity = false;
        sgdlib::FeatureScalarType best_loss = sgdlib::detail::INF;
        sgdlib::FeatureScalarType wscale = 1.0;

        // initialize a lookup table for training X, y
        this->X_data_index_.resize(num_samples);
        std::iota(this->X_data_index_.begin(), this->X_data_index_.end(), 0);

        // init x_i, y_i
        sgdlib::LabelScalarType y;
        std::vector<sgdlib::FeatureScalarType> x(num_features);

        // initialize loss, this->loss_history_, gradient,
        sgdlib::FeatureScalarType l2_penalty;
        sgdlib::FeatureScalarType bias_update;
        sgdlib::FeatureScalarType y_hat, loss, dloss;
        std::vector<sgdlib::FeatureScalarType> weight_update(num_features);
        this->loss_history_.reserve(num_samples * this->max_iters_);

        // initialize w0 (weight) and b0 (bias)
        std::vector<sgdlib::FeatureScalarType> w0 = this->w0_;
        sgdlib::FeatureScalarType b0 = this->b0_;

        // start to loop
        for (iter = 0; iter < this->max_iters_; ++iter) {
            // enable to shuffle mask of data for on batch
            if (this->shuffle_) {
                this->random_state_.shuffle<std::size_t>(this->X_data_index_);
            }

            // apply lr decay policy to compute eta
            sgdlib::FeatureScalarType eta = this->lr_decay_->compute(iter);
            sgdlib::FeatureScalarType eta_alpha = eta * this->alpha_;
            for (std::size_t i = 0; i < step_per_iter; ++i) {
                // reset
                loss = 0.0;
                dloss = 0.0;
                bias_update = 0.0;
                std::fill(weight_update.begin(), weight_update.end(), 0.0);

                // compute start/end index of one batch, use min to avoid less than batch-size
                std::size_t batch_start = i * this->batch_size_;
                std::size_t batch_end = std::min(batch_start + this->batch_size_, num_samples);
                // iterate all rows of batch
                for (std::size_t j = batch_start; j < batch_end; ++j) {
                    std::size_t x_row_index = this->X_data_index_[j];
                    dataset.X_row_data(x_row_index, x);
                    dataset.y_row_data(x_row_index, y);

                    // compute predicted label proba XW + b
                    y_hat = sgdlib::detail::vecdot<sgdlib::FeatureScalarType>(x, w0);
                    y_hat = y_hat * wscale + b0;

                    // evaluate the loss on one row of X, and calculate the derivatives of the loss
                    loss += this->loss_fn_->evaluate(y_hat, y);
                    dloss += this->loss_fn_->derivate(y_hat, y);

                    // clip dloss with large values
                    sgdlib::detail::clip<sgdlib::FeatureScalarType>(dloss, -sgdlib::detail::MAX_DLOSS, sgdlib::detail::MAX_DLOSS);

                    if (dloss != 0.0) {
                        // Scales sample x by constant wscale and add it to weight:
                        // deflation of the sample feature values, adding to weights
                        // means that this scaled sample directly affects the final output
                        dloss /= wscale;
                        sgdlib::detail::vecscale<sgdlib::FeatureScalarType>(x, dloss, weight_update);
                        sgdlib::detail::vecscale<sgdlib::FeatureScalarType>(weight_update, 2.0, weight_update);
                        bias_update += dloss;
                    }
                }

                // scale weight vector by a scalar factor
                wscale *= std::max(0.0, 1.0 - (eta_alpha));
                if (wscale < sgdlib::detail::WSCALE_THRESHOLD) {
                    sgdlib::detail::vecscale<sgdlib::FeatureScalarType>(w0, wscale, w0);
                    wscale = 1.0;
                }

                // compute loss/weight_gradient/bias_gradient for one batch data point
                // weight_update /= batch_size
                // bias_update /= batch_size
                if (this->batch_size_ > 1) {
                    loss *= inv_batch_size;
                    sgdlib::detail::vecscale<sgdlib::FeatureScalarType>(
                        weight_update, inv_batch_size, weight_update
                    );
                    bias_update *= inv_batch_size;
                }

                // add L2 penalty for weight
                if (this->alpha_ > 0.0) {
                    l2_penalty = this->alpha_ *
                        sgdlib::detail::vecnorm2<sgdlib::FeatureScalarType>(w0, true) * inv_num_samples;
                    loss += l2_penalty;
                    sgdlib::detail::vecadd<sgdlib::FeatureScalarType>(
                        w0, weight_update,
                        2.0 * this->alpha_ * inv_num_samples,
                        weight_update
                    );
                }

                // update w0: w = w - lr * dw and b0: b = b - lr * db
                sgdlib::detail::vecdiff<sgdlib::FeatureScalarType>(w0, weight_update, eta, w0);
                b0 -= 2.0 * eta * bias_update;

                // store loss value into this->loss_history_
                this->loss_history_.push_back(loss);
            }

            // ---Convergence test---
            // check under/overflow
            if (sgdlib::detail::hasinf<sgdlib::FeatureScalarType>(w0) ||
                sgdlib::detail::isinf<sgdlib::FeatureScalarType>(b0)) {
                is_infinity = true;
                break;
            }

            // compute sum of the loss value for one full batch
            sgdlib::FeatureScalarType sum_loss = sgdlib::detail::vecaccumul<sgdlib::FeatureScalarType>(
                this->loss_history_.data() + (iter * step_per_iter),
                this->loss_history_.data() + ((iter + 1) * step_per_iter)
            );
            if ((tol_ > -sgdlib::detail::INF) && (sum_loss > best_loss - this->tol_ * num_samples)) {
                no_improvement_count++;
            }
            else {
                no_improvement_count = 0;
            }
            if (sum_loss < best_loss) {
                best_loss = sum_loss;
            }

            // if there is no improvement is bigger than the threshold
            if (no_improvement_count >= this->num_iters_no_change_) {
                if (this->verbose_) {
                    PRINT_RUNTIME_INFO(1, "Convergence after ", iter + 1, " epochs.");
                }
                is_converged = true;
                break;
            }

            if (this->verbose_) {
                PRINT_RUNTIME_INFO(2, "Epoch = ", iter + 1,
                                   ", xnorm2 = ", sgdlib::detail::vecnorm2<sgdlib::FeatureScalarType>(w0, true),
                                   ", avg loss = ", sum_loss / static_cast<sgdlib::FeatureScalarType>(step_per_iter));
            }
        }
        // shrink the this->loss_history_
        this->loss_history_.shrink_to_fit();

        if (callback_) {
            callback_(this->loss_history_);
        }

        if (is_infinity) {
            THROW_RUNTIME_ERROR("Floating-point under-/overflow occurred at epoch ", (iter + 1),
                                ", try to scale input data with standard or minmax.");
        }

        if (!is_converged) {
            THROW_RUNTIME_ERROR("Not converge, current number of epoch ", (iter + 1),
                                ", the batch size ", this->batch_size_,
                                ", try to apply different parameters.");
        }
        this->w_opt_ = w0;
        this->b_opt_ = b0;
    }

};

} // namespace sgdlib

#endif // ALGORITHM_GD_SGD_HPP_
