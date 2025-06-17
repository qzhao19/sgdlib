#ifndef CORE_STEPSIZE_SEARCH_CONSTANT_SEARCH_HPP_
#define CORE_STEPSIZE_SEARCH_CONSTANT_SEARCH_HPP_

#include "base.hpp"

namespace sgdlib {
namespace detail {

class ConstantSearch final: public StepSizeSearch {
public:
    ConstantSearch(const sgdlib::ArrayDatasetType &dataset,
                   const std::shared_ptr<sgdlib::detail::LossFunctionType> &loss_fn,
                   std::shared_ptr<sgdlib::StepSizeSearchParamType> stepsize_search_params): StepSizeSearch (
                        dataset,
                        loss_fn,
                        stepsize_search_params) {};
    ~ConstantSearch() = default;

    /**
     * Compute step size and it is specifically used for the SAG optimizer.
    */
    int search(bool is_saga, sgdlib::ScalarType &step_size) override {
        std::size_t num_samples = this->dataset_.nrows();

        std::vector<sgdlib::FeatureScalarType> X_row_norm(num_samples);
        sgdlib::detail::row_norms<sgdlib::FeatureScalarType>(this->dataset_, true, X_row_norm);

        sgdlib::FeatureScalarType max_sum = *std::max_element(X_row_norm.begin(), X_row_norm.end());

        sgdlib::ScalarType alpha_scaled = this->stepsize_search_params_->alpha / num_samples;
        sgdlib::ScalarType L = 0.25 * (max_sum + 1.0) + alpha_scaled;

        if (is_saga) {
            sgdlib::ScalarType mu = std::min(2 * num_samples * alpha_scaled, L);
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
