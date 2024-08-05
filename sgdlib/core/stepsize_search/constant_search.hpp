#ifndef CORE_STEPSIZE_SEARCH_CONSTANT_EARCH_HPP_
#define CORE_STEPSIZE_SEARCH_CONSTANT_EARCH_HPP_

#include "base.hpp"

namespace sgdlib {

template <typename LossFuncType>
class ConstantSearch final: public StepSizeSearch<LossFuncType>{
public:
    ConstantSearch(const std::vector<FeatureType>& X, 
                   const std::vector<LabelType>& y,
                   const std::shared_ptr<LossFuncType>& loss_fn,
                   StepSizeSearchParamType stepsize_search_params): StepSizeSearch<LossFuncType>(
                        X, y, 
                        loss_fn, 
                        stepsize_search_params) {};
    ~ConstantSearch() {};

    /** 
     * Compute step size and it is specifically used for the SAG optimizer.
    */
    int search(bool is_saga, double& step_size) override {
        std::size_t num_samples = this->y_.size();

        std::vector<FeatureType> X_row_norm(num_samples);
        sgdlib::internal::row_norms<FeatureType>(this->X_, true, X_row_norm);

        FeatureType max_sum = *std::max_element(X_row_norm.begin(), X_row_norm.end());

        double alpha_scaled = this->stepsize_search_params_["alpha"] / num_samples;
        double L = 0.25 * (max_sum + 1.0) + alpha_scaled;

        if (is_saga) {
            double mu = std::min(2 * num_samples * alpha_scaled, L);
            step_size = 1.0 / (2* L + mu);
        }
        else {
            step_size =  1.0 / L;
        }

        return 0;
    }
};

} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_CONSTANT_EARCH_HPP_