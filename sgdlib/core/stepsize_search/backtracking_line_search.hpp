#ifndef CORE_STEPSIZE_SEARCH_BACKTRACKING_LINE_SEARCH_HPP_
#define CORE_STEPSIZE_SEARCH_BACKTRACKING_LINE_SEARCH_HPP_

#include "base.hpp"

namespace sgdlib {
namespace detail {

class BacktrackingLineSearch final: public StepSizeSearch {
public:
    BacktrackingLineSearch(const sgdlib::ArrayDatasetType &dataset,
                           const std::shared_ptr<sgdlib::detail::LossFunctionType> &loss_fn,
                           std::shared_ptr<sgdlib::StepSizeSearchParamType> stepsize_search_params): StepSizeSearch(
                                dataset,
                                loss_fn,
                                stepsize_search_params) { };
    ~BacktrackingLineSearch() = default;

    int search(const std::vector<sgdlib::FeatureScalarType> &xp,
               const std::vector<sgdlib::FeatureScalarType> &gp,
               const std::vector<sgdlib::FeatureScalarType> &d,
               std::vector<sgdlib::FeatureScalarType> &x,
               std::vector<sgdlib::FeatureScalarType> &g,
               sgdlib::FeatureScalarType &fx,
               sgdlib::ScalarType &stepsize) override {

        std::size_t num_samples = this->dataset_.nrows();
        std::size_t num_features = this->dataset_.ncols();
        sgdlib::ScalarType dec_factor = this->stepsize_search_params_->dec_factor;
        sgdlib::ScalarType inc_factor = this->stepsize_search_params_->inc_factor;

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
        sgdlib::ScalarType width;
        // define loss and grad vector
        sgdlib::FeatureScalarType loss;
        std::vector<sgdlib::FeatureScalarType> grad(num_features);

        int count = 0;
        while (true) {
            // x_{k+1} = x_k + stepsize * d_k
            sgdlib::detail::vecadd<sgdlib::FeatureScalarType>(d, xp, stepsize, x);

            // reset gradient vector for ecah loop
            std::memset(grad.data(), 0, num_features * sizeof(sgdlib::FeatureScalarType));

            // compute the loss value and gradient vector
            loss = this->loss_fn_->evaluate_with_gradient(this->dataset_, x, grad);

            fx = loss;
            g = grad;

            // increment
            ++count;

            if (fx > fx_init + stepsize * dg_test) {
                width = dec_factor;
            }
            else {
                // check the armijo condition
                if (this->stepsize_search_params_->condition == "ARMIJO") {
                    return count;
                }
                sgdlib::ScalarType dg = sgdlib::detail::vecdot<sgdlib::ScalarType>(d, g);
                if (dg < this->stepsize_search_params_->wolfe * dg_init) {
                    width = inc_factor;
                }
                else {
                    if (this->stepsize_search_params_->condition == "WOLFE") {
                        return count;
                    }

                    if (dg > (-this->stepsize_search_params_->wolfe * dg_init)) {
                        width = dec_factor;
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
            stepsize *= width;
        }
    }

};

} // namespace detail
} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_BACKTRACKING_LINE_SEARCH_HPP_
