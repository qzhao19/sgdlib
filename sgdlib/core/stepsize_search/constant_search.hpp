#ifndef CORE_STEPSIZE_SEARCH_CONSTANT_SEARCH_HPP_
#define CORE_STEPSIZE_SEARCH_CONSTANT_SEARCH_HPP_

#include "base.hpp"

namespace sgdlib {
namespace detail {

template <typename LossFuncType>
class ConstantSearch final: public StepSizeSearch<LossFuncType>{
public:
    ConstantSearch(const std::vector<FeatValType>& X, 
                   const std::vector<LabelValType>& y,
                   const std::shared_ptr<LossFuncType>& loss_fn,
                   StepSizeSearchParamType* stepsize_search_params): StepSizeSearch<LossFuncType>(
                        X, y, 
                        loss_fn, 
                        stepsize_search_params) {};
    ~ConstantSearch() = default;

    /** 
     * Compute step size and it is specifically used for the SAG optimizer.
    */
    int search(bool is_saga, FloatType& step_size) override {
        std::size_t num_samples = this->y_.size();

        std::vector<FeatValType> X_row_norm(num_samples);
        sgdlib::detail::row_norms<FeatValType>(this->X_, true, X_row_norm);

        FeatValType max_sum = *std::max_element(X_row_norm.begin(), X_row_norm.end());

        FloatType alpha_scaled = this->stepsize_search_params_->alpha / num_samples;
        FloatType L = 0.25 * (max_sum + 1.0) + alpha_scaled;

        if (is_saga) {
            FloatType mu = std::min(2 * num_samples * alpha_scaled, L);
            step_size = 1.0 / (2* L + mu);
        }
        else {
            step_size =  1.0 / L;
        }

        return 0;
    }
};

} // namespace detail
} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_CONSTANT_SEARCH_HPP_