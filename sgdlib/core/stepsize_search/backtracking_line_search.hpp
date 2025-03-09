#ifndef CORE_STEPSIZE_SEARCH_BACKTRACKING_LINE_SEARCH_HPP_
#define CORE_STEPSIZE_SEARCH_BACKTRACKING_LINE_SEARCH_HPP_

#include "base.hpp"

namespace sgdlib {

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
               std::vector<FeatValType>& x,
               std::vector<FeatValType>& g, 
               std::vector<FeatValType>& d,
               FeatValType& fx,
               FloatType& stepsize) {
        
        num_features_ = x.size();
        FeatValType y_hat;
        FloatType dec_factor = this->stepsize_search_params_->dec_factor;
        FloatType inc_factor = this->stepsize_search_params_->inc_factor;
        
        if (stepsize <= 0.0) {
            // step must be positive
            return LBFGS_ERROR_INVALID_PARAMETERS;
        }

        // initialize fx_init and compute init gradient in search direction
        FeatValType fx_init = fx;
        FeatValType dg_init = 0.0; 
        sgdlib::internal::dot<FeatValType>(d, g, dg_init);

        if (dg_init > 0.0) {
            // moving direction increases the objective function value
            return LBFGS_ERROR_INCREASE_GRADIENT;
        }

        FloatType dg_test = this->stepsize_search_params_->ftol * dg_init;
        FloatType width;
        int count = 0;

        while (true) {
            // x_{k+1} = x_k + stepsize * d_k
            sgdlib::internal::dot<FeatValType>(d, stepsize);
            sgdlib::internal::add<FeatValType>(xp, d, x);
            
            // compute the loss value and gradient vector
            for (std::size_t i = 0; i < num_samples_; ++i) {
                y_hat = std::inner_product(&this->X[i * num_features_], 
                                           &this->X[(i + 1) * num_features_], 
                                           x.begin(), 0.0);
                fx += this->loss_fn_->evaluate(y_hat, this->y[i]);
                for (std::size_t j = 0; j < num_features_; ++j) {
                    g[j] += this->loss_fn_->derivate(y_hat, this->y[i]) * this->X[i * num_features_ + j];
                }
            }
            fx /= static_cast<FeatValType>(num_samples_);
            std::transform(g.begin(), g.end(), g.begin(),
                          [this](FeatValType val) { 
                            return val / static_cast<FeatValType>(num_samples_); 
                        });
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
                sgdlib::internal::dot<FloatType>(d, g, dg);
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

            if (stepsize < this->linesearch_params_->min_stepsize) {
                return LBFGS_ERROR_MINIMUM_STEPSIZE;
            }

            if (stepsize > this->linesearch_params_->max_stepsize) {
                return LBFGS_ERROR_MAXIMUM_STEPSIZE;
            }

            if (count >= this->linesearch_params_->max_searches) {
                return LBFGS_ERROR_MAXIMUM_SEARCHES;
            }
            stepsize *= width;
        }
    }

};

} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_BACKTRACKING_LINE_SEARCH_HPP_