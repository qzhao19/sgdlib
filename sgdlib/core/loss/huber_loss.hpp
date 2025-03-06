#ifndef CORE_LOSS_HUBER_LOSS_HPP_
#define CORE_LOSS_HUBER_LOSS_HPP_

#include "base.hpp"

namespace sgdlib {

/** 
 * @file huber_loss.hpp
 * 
 * @brief Huber loss function
*/
class HuberLoss final: public LossFunction {
public:
    HuberLoss(LossParamType loss_param): LossFunction(loss_param) {};
    ~HuberLoss() {};

    virtual FeatValType evaluate(const FeatValType& y_pred, 
                                 const LabelValType& y_true) const override {
        
        FeatValType z = y_pred * static_cast<FeatValType>(y_true);
        if (z > 18.0) {
            return std::exp(-z);
        }
        if (z < -18.0) {
            return -z;
        }
        return std::log(1.0 + std::exp(-z));
    }

    virtual FeatValType derivate(const FeatValType& y_pred, 
                                 const LabelValType& y_true) const override {

        FeatValType z = y_pred * static_cast<FeatValType>(y_true);
        if (z > 18.0) {
            return std::exp(-z) * (-static_cast<FeatValType>(y_true));
        }
        if (z < -18.0) {
            return -static_cast<FeatValType>(y_true);
        }
        return -static_cast<FeatValType>(y_true) / (std::exp(z) + 1.0);
    }
    
};

} // namespace sgdlib

#endif // CORE_LOSS_HUBER_LOSS_HPP_