#ifndef CORE_STEPSIZE_SEARCH_BASIC_LINE_EARCH_HPP_
#define CORE_STEPSIZE_SEARCH_BASIC_LINE_EARCH_HPP_

#include "base.hpp"

namespace sgdlib {

template <typename LossFuncType>
class BasicLineSearch final: public StepSizeSearch<LossFuncType>{
public:
    BasicLineSearch(const std::vector<FeatureType>& X, 
                    const std::vector<LabelType>& y,
                    StepSizeSearchParamType stepsize_search_params, 
                    std::unique_ptr<LossFuncType> loss_fn): StepSizeSearch<LossFuncType>(
                        X, y, stepsize_search_params, loss_fn) {
        std::size_t num_samples = y.size();
        this->lipschitz_ = 1.0 / this->stepsize_search_params_["eta0"] - this->stepsize_search_params_["alpha"];
        this->linesearch_scaling_ = std::pow(2.0, this->stepsize_search_params_["max_searches"] / num_samples);

    };
    ~BasicLineSearch() {};

    /** 
     * Compute step size and it is specifically used for the SAG optimizer.
    */
    int search(const FeatureType& y_pred, 
               const LabelType& y_true, 
               const FeatureType& grad,
               const FeatureType& xnorm, 
               const std::size_t& step,
               double& stepsize) override {
        
        bool is_valid;
        FeatureType a, b;
        if (step % this->stepsize_search_params_["max_searches"] == 0) {
            for (size_t i = 0; i < this->stepsize_search_params_["max_iters"]; ++i) {
                a = this->loss_fn_(y_pred - grad * xnorm / this->lipschitz_, y_true);
                b = this->loss_fn_(y_pred, y_true) - 0.5 * grad * grad * xnorm / this->lipschitz_;
                if (a <= b) {
                    this->lipschitz_ /= this->linesearch_scaling_;
                    is_valid = true;
                    break;
                }
                else {
                    this->lipschitz_ *= 2.0;
                    is_valid = false;
                }
            }

            if (!is_valid) {
                return -1;
            }
            stepsize = 1.0 / (this->lipschitz_ + alpha);
        }
        return 0;
    }
};

} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_BASIC_LINE_EARCH_HPP_