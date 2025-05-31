#ifndef CORE_STEPSIZE_SEARCH_BACKTRACKING_LINE_SEARCH_HPP_
#define CORE_STEPSIZE_SEARCH_BACKTRACKING_LINE_SEARCH_HPP_

#include "base.hpp"

namespace sgdlib {
namespace detail {

template <typename LossFuncType>
class BacktrackingLineSearch final: public StepSizeSearch<LossFuncType> {
protected:
    std::size_t num_samples_;
    std::size_t num_features_;

public:
    BacktrackingLineSearch(const std::vector<FeatValType>& X,
                           const std::vector<LabelValType>& y,
                           const std::shared_ptr<LossFuncType>& loss_fn,
                           StepSizeSearchParamType* stepsize_search_params): StepSizeSearch<LossFuncType>(
                                X, y,
                                loss_fn,
                                stepsize_search_params) {
        num_samples_ = y.size();
    };
    ~BacktrackingLineSearch() = default;

    int search(const std::vector<FeatValType>& xp,
               const std::vector<FeatValType>& gp,
               const std::vector<FeatValType>& d,
               std::vector<FeatValType>& x,
               std::vector<FeatValType>& g,
               FeatValType& fx,
               FloatType& stepsize) override {

        num_features_ = x.size();
        const FeatValType inv_num_samples = 1.0 / static_cast<FeatValType>(num_samples_);

        FloatType dec_factor = this->stepsize_search_params_->dec_factor;
        FloatType inc_factor = this->stepsize_search_params_->inc_factor;

        if (stepsize <= 0.0) {
            // step must be positive
            return LBFGS_ERROR_INVALID_PARAMETERS;
        }

        // initialize fx_init and compute init gradient in search direction
        FeatValType fx_init = fx;
        FeatValType dg_init;
        sgdlib::detail::dot<FeatValType>(d, g, dg_init);

        if (dg_init > 0.0) {
            // moving direction increases the objective function value
            return LBFGS_ERROR_INCREASE_GRADIENT;
        }

        FloatType dg_test = this->stepsize_search_params_->ftol * dg_init;
        FloatType width;
        // define loss and grad vector
        FeatValType loss;
        std::vector<FeatValType> grad(num_features_);

        int count = 0;
        while (true) {
            // x_{k+1} = x_k + stepsize * d_k
            sgdlib::detail::vecadd<FeatValType>(d, xp, stepsize, x);

            // reset gradient vector for ecah loop
            std::memset(grad.data(), 0, grad.size() * sizeof(FeatValType));

            // compute the loss value and gradient vector
            loss = this->loss_fn_->evaluate_with_gradient(this->X_, this->y_, x, grad);
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
                FloatType dg;
                sgdlib::detail::dot<FloatType>(d, g, dg);
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
