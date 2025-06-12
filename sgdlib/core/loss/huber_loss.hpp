#ifndef CORE_LOSS_HUBER_LOSS_HPP_
#define CORE_LOSS_HUBER_LOSS_HPP_

#include "base.hpp"

namespace sgdlib {
namespace detail {

/**
 * @brief Huber loss function
*/
class HuberLoss final: public LossFunction {
public:
    HuberLoss(LossParamType loss_param): LossFunction(loss_param) {};
    ~HuberLoss() = default;

    FeatValType evaluate(const FeatValType &y_pred,
                         const LabelValType &y_true) const override {
        return 0.0;
    }

    FeatValType derivate(const FeatValType &y_pred,
                         const LabelValType &y_true) const override {
        return 0.0;
    }

    FeatValType evaluate_with_gradient(const sgdlib::detail::ArrayDatasetType &dataset,
                                       const std::vector<FeatValType> &w,
                                       std::vector<FeatValType> &grad) const override {
        return 0.0;
    }

};

} // namespace detail
} // namespace sgdlib

#endif // CORE_LOSS_HUBER_LOSS_HPP_
