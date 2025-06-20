#ifndef CORE_STEPSIZE_SEARCH_BASIC_LINE_SEARCH_HPP_
#define CORE_STEPSIZE_SEARCH_BASIC_LINE_SEARCH_HPP_

#include "base.hpp"

namespace sgdlib {
namespace detail {

class ExactLineSearch final: public StepSizeSearch {
public:
    ExactLineSearch(const sgdlib::ArrayDatasetType &dataset,
                    const std::shared_ptr<sgdlib::detail::LossFunctionType> &loss_fn,
                    std::shared_ptr<sgdlib::StepSizeSearchParamType> stepsize_search_params): StepSizeSearch (
                        dataset,
                        loss_fn,
                        stepsize_search_params) {
        std::size_t num_samples = this->dataset_.nrows();
        this->lipschitz_ = 1.0 / this->stepsize_search_params_->eta0 - this->stepsize_search_params_->alpha;
        this->linesearch_scaling_ = std::pow(2.0, static_cast<sgdlib::ScalarType>(this->stepsize_search_params_->max_searches) / num_samples);
    };
    ~ExactLineSearch() = default;

    /**
     * Compute step size with basic line search and it is specifically used for the SAG optimizer.
    */
    int search(const sgdlib::FeatureScalarType &y_pred,
               const sgdlib::LabelScalarType &y_true,
               const sgdlib::FeatureScalarType &grad,
               const sgdlib::FeatureScalarType &xnorm,
               const std::size_t &step,
               sgdlib::ScalarType &stepsize) override {
        bool is_valid;
        sgdlib::FeatureScalarType a, b;

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

} // namespace detail
} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_BASIC_LINE_SEARCH_HPP_
