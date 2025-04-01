#ifndef CORE_LOSS_HUBER_LOSS_HPP_
#define CORE_LOSS_HUBER_LOSS_HPP_

#include "base.hpp"

namespace sgdlib {
namespace detail {

/** 
 * @file huber_loss.hpp
 * 
 * @brief Huber loss function
*/
class HuberLoss final: public LossFunction {
public:
    HuberLoss(LossParamType loss_param): LossFunction(loss_param) {};
    ~HuberLoss() = default;

    FeatValType evaluate(const FeatValType& y_pred, 
                         const LabelValType& y_true) const override {
        return 0.0;
    }

    FeatValType derivate(const FeatValType& y_pred, 
                         const LabelValType& y_true) const override {
        return 0.0;
    }
    
};

} // namespace detail
} // namespace sgdlib

#endif // CORE_LOSS_HUBER_LOSS_HPP_