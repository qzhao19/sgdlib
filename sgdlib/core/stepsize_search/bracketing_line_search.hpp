#ifndef CORE_STEPSIZE_SEARCH_BRACKETING_LINE_SEARCH_HPP_
#define CORE_STEPSIZE_SEARCH_BRACKETING_LINE_SEARCH_HPP_

#include "base.hpp"

namespace sgdlib {
namespace detail {

template <typename LossFuncType>
class BracketingLineSearch final: public StepSizeSearch<LossFuncType> {
public:
    BracketingLineSearch(const sgdlib::detail::ArrayDatasetType &dataset,
                         const std::shared_ptr<LossFuncType> &loss_fn,
                         std::shared_ptr<StepSizeSearchParamType> stepsize_search_params): StepSizeSearch<LossFuncType>(
                                dataset,
                                loss_fn,
                                stepsize_search_params) { };
    ~BracketingLineSearch() = default;

    int search(const std::vector<FeatValType> &xp,
               const std::vector<FeatValType> &gp,
               const std::vector<FeatValType> &d,
               std::vector<FeatValType> &x,
               std::vector<FeatValType> &g,
               FeatValType &fx,
               FloatType &stepsize) override {

        // num_features_ = x.size();
        std::size_t num_features = this->dataset_.ncols();

        FloatType dec_factor = this->stepsize_search_params_->dec_factor;
        FloatType inc_factor = this->stepsize_search_params_->inc_factor;

        if (stepsize <= 0.0) {
            // step must be positive
            return LBFGS_ERROR_INVALID_PARAMETERS;
        }

        // initialize fx_init and compute init gradient in search direction
        FeatValType fx_init = fx;
        FeatValType dg_init = sgdlib::detail::vecdot<FeatValType>(d, g);

        if (dg_init > 0.0) {
            // moving direction increases the objective function value
            return LBFGS_ERROR_INCREASE_GRADIENT;
        }

        FloatType dg_test = this->stepsize_search_params_->ftol * dg_init;
        FloatType stepsize_hi = INF;
        FloatType stepsize_lo = 0.0;
        // define loss and grad vector
        FeatValType loss;
        std::vector<FeatValType> grad(num_features, 0.0);

        int count = 0;
        while (true) {
            // x_{k+1} = x_k + stepsize * d_k
            sgdlib::detail::vecadd<FeatValType>(d, xp, stepsize, x);

            // reset gradient vector for ecah loop
            std::memset(grad.data(), 0, num_features * sizeof(FeatValType));

            // compute the loss value and gradient vector
            loss = this->loss_fn_->evaluate_with_gradient(this->dataset_, x, grad);
            fx = loss;
            g = grad;

            ++count;

            if (fx > fx_init + stepsize * dg_test) {
                stepsize_hi = stepsize;
            }
            else {
                // check the armijo condition
                if (this->stepsize_search_params_->condition == "ARMIJO") {
                    return count;
                }
                FloatType dg = sgdlib::detail::vecdot<FloatType>(d, g);
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
