#ifndef ALGORITHM_GD_SVRG_HPP_
#define ALGORITHM_GD_SVRG_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

class SVRG: public Optimizer {
public:
    SVRG(const std::vector<sgdlib::FeatureScalarType>& w0,
        std::string loss,
        std::string lr_policy,
        sgdlib::ScalarType alpha,
        sgdlib::ScalarType eta0,
        sgdlib::ScalarType tol,
        sgdlib::ScalarType gamma,
        std::size_t max_iters,
        std::size_t num_inner,
        std::size_t random_seed,
        bool shuffle = true,
        bool verbose = true): Optimizer(w0, loss, tol, max_iters, verbose) {
            this->lr_policy_ = lr_policy;
            this->alpha_ = alpha;
            this->eta0_ = eta0;
            this->gamma_ = gamma;
            this->num_inner_ = num_inner;
            this->random_seed_ = random_seed;
            this->shuffle_ = shuffle;
            this->init_random_state();
            this->init_loss_params();
        }

    ~SVRG() = default;

    void optimize(const sgdlib::ArrayDatasetType& dataset) override {
        // Get the number of samples and features from the input data
        const std::size_t num_samples = dataset.nrows();
        const std::size_t num_features = dataset.ncols();
        const sgdlib::FeatureScalarType inv_num_samples = 1.0 / static_cast<sgdlib::FeatureScalarType>(num_samples);
        // Initialize iteration counter, sample index, and batch size
        std::size_t iter = 0, sample_index = 0;

        // Initialize convergence flag and best loss
        bool is_converged = false;
        bool is_infinity = false;
        sgdlib::FeatureScalarType wscale = 1.0;

        const sgdlib::FeatureScalarType eta_avg = this->eta0_ * inv_num_samples;
        const sgdlib::FeatureScalarType eta_alpha = this->eta0_ * this->alpha_;

        //
        sgdlib::FeatureScalarType reg_dw, fnorm, fnorm_ratio, init_fnorm, alpha_scaled;

        // init x_i, y_i
        sgdlib::LabelScalarType y;
        std::vector<sgdlib::FeatureScalarType> x(num_features);

        // initialize a lookup table for training X, y
        this->X_data_index_.resize(num_samples);
        std::iota(this->X_data_index_.begin(), this->X_data_index_.end(), 0);
        this->loss_history_.reserve(this->max_iters_ * this->num_inner_);

        // initialize gradient,
        sgdlib::FeatureScalarType y_hat, grad, loss, total_loss;
        std::vector<sgdlib::FeatureScalarType> grad_history(num_samples, 0.0);
        std::vector<sgdlib::FeatureScalarType> full_weight_update(num_features, 0.0);
        std::vector<std::size_t> update_history(num_features, 0);

        // initialize w0 (weight)
        std::vector<sgdlib::FeatureScalarType> w0 = this->w0_;

        // start iteration
        for (iter = 0; iter < this->max_iters_; ++iter) {
            // Reset the full weight update vector to zero at the beginning of each iteration
            sgdlib::detail::vecset<sgdlib::FeatureScalarType>(full_weight_update, 0.0);

            // trigger callback function to get grad_history
            this->loss_fn_->set_callback([&grad_history](const std::vector<sgdlib::FeatureScalarType>& dloss_history){
                grad_history.assign(dloss_history.begin(), dloss_history.end());
            });
            total_loss = this->loss_fn_->evaluate_with_gradient(dataset, w0, full_weight_update);

            // inner loop
            for (std::size_t n = 0; n < this->num_inner_; ++n) {
                // shuffle the samples
                if (this->shuffle_) {
                    sample_index = this->random_state_.sample<std::size_t>(this->X_data_index_);
                }
                else {
                    sample_index = n % num_samples;
                }

                // get X_i and y_i
                dataset.X_row_data(sample_index, x);
                dataset.y_row_data(sample_index, y);

                // just-in-time update for full weight 1/n * sum(d(f_k_w))
                // n - update_history[j]: weight is not be updated for n - update_history[j] times
                // need to compensate the full weight update
                // update_history[j] = n: weight is already updated now
                // Update the weights to compensate for the full gradient
                if (n > 0) {
                    for (std::size_t j = 0; j < num_features; ++j) {
                        w0[j] -= eta_avg / wscale * (n - update_history[j]) * full_weight_update[j];
                        update_history[j] = n;
                    }
                }

                // current gradient at sample_index row data
                // y_hat = sgdlib::detail::vecdot<sgdlib::FeatureScalarType>(
                //     X.data() + (sample_index * num_features),
                //     X.data() + ((sample_index + 1) * num_features),
                //     w0.data()
                // );
                y_hat = sgdlib::detail::vecdot<sgdlib::FeatureScalarType>(x, w0);

                y_hat = y_hat * wscale;
                // loss = this->loss_fn_->evaluate(y_hat, y[sample_index]);
                // grad = this->loss_fn_->derivate(y_hat, y[sample_index]);

                loss = this->loss_fn_->evaluate(y_hat, y);
                grad = this->loss_fn_->derivate(y_hat, y);

                // update the weight scale factor based on eta*alpha
                wscale *= (1 - eta_alpha);

                // update w: old_gradient - new_gradient
                // sgdlib::detail::vecadd<sgdlib::FeatureScalarType>(
                //     X.data() + (sample_index * num_features),
                //     X.data() + ((sample_index + 1) * num_features),
                //     this->eta0_ * (grad_history[sample_index] - grad) / wscale, w0
                // );

                sgdlib::detail::vecadd<sgdlib::FeatureScalarType>(
                    x, this->eta0_ * (grad_history[sample_index] - grad) / wscale, w0
                );

                // check if the weight scale factor is below the threshold
                // rescale the weights and reset the scale factor
                if (wscale < sgdlib::detail::WSCALE_THRESHOLD) {
                    sgdlib::detail::vecscale<sgdlib::FeatureScalarType>(w0, wscale, w0);
                    wscale = 1.0;
                }

                // store loss value into this->loss_history_
                this->loss_history_.push_back(loss);
            }

            // finalize
            for (std::size_t j = 0; j < num_features; ++j) {
                w0[j] -= eta_avg / wscale * (this->num_inner_ - update_history[j]) * full_weight_update[j];
                update_history[j] = 0;
            }

            // compute the L2 norm of the loss function
            // fnorm = sqrt[(w0 * alpha + 1/n * d(f_k_w))**2]
            fnorm = 0.0;
            alpha_scaled = this->alpha_ * wscale;
            for (std::size_t j = 0; j < num_features; ++j) {
                reg_dw = full_weight_update[j] * inv_num_samples + alpha_scaled * w0[j];
                fnorm += reg_dw * reg_dw;
            }
            fnorm = std::sqrt(fnorm);

            // Store the initial gradient norm
            if (iter == 0) {
                init_fnorm = fnorm;
            }

            // Compute the ratio of the current gradient norm to the previous one
            fnorm_ratio = fnorm / init_fnorm;

            if (this->verbose_) {
                PRINT_RUNTIME_INFO(1, "Epoch = ", iter + 1, ", fnorm_ratio = ", fnorm_ratio);
            }

            if (fnorm_ratio <= this->tol_) {
                is_converged = true;
                break;
            }

            if (sgdlib::detail::hasinf<sgdlib::FeatureScalarType>(w0)) {
                is_infinity = true;
                break;
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
                                ", try apply different parameters.");
        }
        this->w_opt_ = w0;
    }

    const sgdlib::FeatureScalarType get_intercept() const override {
        THROW_LOGIC_ERROR("The 'get_intercept' method is not supported for this SVRG optimizer.");
        return 0.0;
    }
};

}

#endif // ALGORITHM_GD_SVRG_HPP_
