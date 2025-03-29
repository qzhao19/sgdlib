#ifndef CORE_STEPSIZE_SEARCH_BASIC_LINE_SEARCH_HPP_
#define CORE_STEPSIZE_SEARCH_BASIC_LINE_SEARCH_HPP_

#include "base.hpp"

namespace sgdlib {

template <typename LossFuncType>
class BasicLineSearch final: public StepSizeSearch<LossFuncType> {
public:
    BasicLineSearch(const std::vector<FeatValType>& X, 
                    const std::vector<LabelValType>& y,
                    const std::shared_ptr<LossFuncType>& loss_fn,
                    StepSizeSearchParamType* stepsize_search_params): StepSizeSearch<LossFuncType>(
                        X, y, 
                        loss_fn, 
                        stepsize_search_params) {
        std::size_t num_samples = y.size();
        this->lipschitz_ = 1.0 / this->stepsize_search_params_->eta0 - this->stepsize_search_params_->alpha;
        this->linesearch_scaling_ = std::pow(2.0, static_cast<FloatType>(this->stepsize_search_params_->max_searches) / num_samples);
    };
    ~BasicLineSearch() = default;

    /** 
     * Compute step size with basic line search and it is specifically used for the SAG optimizer.
    */
    int search(const FeatValType& y_pred, 
               const LabelValType& y_true, 
               const FeatValType& grad,
               const FeatValType& xnorm, 
               const std::size_t& step,
               FloatType& stepsize) override {
        bool is_valid;
        FeatValType a, b;

        if ((step % this->stepsize_search_params_->max_searches == 0) && (std::abs(grad) > 1e-8)) {
            for (std::size_t i = 0; i < this->stepsize_search_params_->max_iters; ++i) {
                a = this->loss_fn_->evaluate(y_pred - grad * xnorm / this->lipschitz_, y_true);
                b = this->loss_fn_->evaluate(y_pred, y_true) - 0.5 * grad * grad * xnorm / this->lipschitz_;
                
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
            stepsize = 1.0 / (this->lipschitz_ + this->stepsize_search_params_->alpha);
        }
        return 0;
    }
};

} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_BASIC_LINE_SEARCH_HPP_