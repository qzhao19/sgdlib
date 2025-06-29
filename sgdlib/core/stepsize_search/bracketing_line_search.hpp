#ifndef CORE_STEPSIZE_SEARCH_BRACKETING_LINE_SEARCH_HPP_
#define CORE_STEPSIZE_SEARCH_BRACKETING_LINE_SEARCH_HPP_

#include "base.hpp"

namespace sgdlib {
namespace detail {

class BracketingLineSearch final: public StepSizeSearch {
public:
    BracketingLineSearch(const sgdlib::ArrayDatasetType &dataset,
                         const std::shared_ptr<sgdlib::detail::LossFunctionType> &loss_fn,
                         std::shared_ptr<sgdlib::StepSizeSearchParamType> stepsize_search_params): StepSizeSearch(
                                dataset,
                                loss_fn,
                                stepsize_search_params) { };
    ~BracketingLineSearch() = default;

    int search(const std::vector<sgdlib::FeatureScalarType> &xp,
               const std::vector<sgdlib::FeatureScalarType> &gp,
               const std::vector<sgdlib::FeatureScalarType> &d,
               std::vector<sgdlib::FeatureScalarType> &x,
               std::vector<sgdlib::FeatureScalarType> &g,
               sgdlib::FeatureScalarType &fx,
               sgdlib::ScalarType &stepsize) override {

        // num_features_ = x.size();
        std::size_t num_samples = this->dataset_.nrows();
        std::size_t num_features = this->dataset_.ncols();
        sgdlib::ScalarType dec_factor = this->stepsize_search_params_->dec_factor;
        sgdlib::ScalarType inc_factor = this->stepsize_search_params_->inc_factor;
        sgdlib::FeatureScalarType inv_num_samples = 1.0 / static_cast<sgdlib::FeatureScalarType>(num_samples);
        if (stepsize <= 0.0) {
            // step must be positive
            return LBFGS_ERROR_INVALID_PARAMETERS;
        }

        // initialize fx_init and compute init gradient in search direction
        sgdlib::FeatureScalarType fx_init = fx;
        sgdlib::FeatureScalarType dg_init = sgdlib::detail::vecdot<sgdlib::FeatureScalarType>(d, g);

        if (dg_init > 0.0) {
            // moving direction increases the objective function value
            return LBFGS_ERROR_INCREASE_GRADIENT;
        }

        sgdlib::ScalarType dg_test = this->stepsize_search_params_->ftol * dg_init;
        sgdlib::ScalarType stepsize_hi = INF;
        sgdlib::ScalarType stepsize_lo = 0.0;
        // define loss and grad vector
        sgdlib::FeatureScalarType total_loss;
        std::vector<sgdlib::FeatureScalarType> total_grad(num_features);

        int count = 0;
        while (true) {
            // x_{k+1} = x_k + stepsize * d_k
            sgdlib::detail::vecadd<sgdlib::FeatureScalarType>(d, xp, stepsize, x);

            // reset gradient vector for ecah loop
            std::memset(total_grad.data(), 0, num_features * sizeof(sgdlib::FeatureScalarType));

            // compute the loss value and gradient vector
            total_loss = this->loss_fn_->evaluate_with_gradient(this->dataset_, x, total_grad);
            total_loss *= inv_num_samples;
            sgdlib::detail::vecscale<sgdlib::FeatureScalarType>(total_grad, inv_num_samples, total_grad);
            fx = total_loss;
            g = total_grad;

            ++count;

            if (fx > fx_init + stepsize * dg_test) {
                stepsize_hi = stepsize;
            }
            else {
                // check the armijo condition
                if (this->stepsize_search_params_->condition == "ARMIJO") {
                    return count;
                }
                sgdlib::ScalarType dg = sgdlib::detail::vecdot<sgdlib::ScalarType>(d, g);
                if (dg < this->stepsize_search_params_->wolfe * dg_init) {
                    stepsize_lo = stepsize;
                }
                else {
                    if (this->stepsize_search_params_->condition == "WOLFE") {
                        return count;
                    }

                    if (dg > (-this->stepsize_search_params_->wolfe * dg_init)) {
                        stepsize_hi = stepsize;
                    }
                    else {
                        return count;
                    }
                }
            }

            if (stepsize < this->stepsize_search_params_->min_stepsize) {
                return LBFGS_ERROR_MINIMUM_STEPSIZE;
            }

            if (stepsize > this->stepsize_search_params_->max_stepsize) {
                return LBFGS_ERROR_MAXIMUM_STEPSIZE;
            }

            if (count >= this->stepsize_search_params_->max_searches) {
                return LBFGS_ERROR_MAXIMUM_SEARCHES;
            }
            stepsize = std::isinf(stepsize_hi) ? 2.0 * stepsize : (stepsize_lo + stepsize_hi) / 2.0;;
        }
    }
};

} // namespace detail
} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_BRACKETING_LINE_SEARCH_HPP_
