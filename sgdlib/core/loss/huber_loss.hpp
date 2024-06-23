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

    virtual FeatureType evaluate(const FeatureType& y_pred, 
                                 const LabelType& y_true) const override {
        
        FeatureType z = y_pred * static_cast<FeatureType>(y_true);
        if (z > 18.0) {
            return std::exp(-z);
        }
        if (z < -18.0) {
            return -z;
        }
        return std::log(1.0 + std::exp(-z));
    }

    virtual FeatureType derivate(const FeatureType& y_pred, 
                                 const LabelType& y_true) const override {

        FeatureType z = y_pred * static_cast<FeatureType>(y_true);
        if (z > 18.0) {
            return std::exp(-z) * (-static_cast<FeatureType>(y_true));
        }
        if (z < -18.0) {
            return -static_cast<FeatureType>(y_true);
        }
        return -static_cast<FeatureType>(y_true) / (std::exp(z) + 1.0);
    }
    
};

} // namespace sgdlib

#endif // CORE_LOSS_HUBER_LOSS_HPP_