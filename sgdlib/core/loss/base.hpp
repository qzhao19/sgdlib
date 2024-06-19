#ifndef CORE_LOSS_BASE_HPP_
#define CORE_LOSS_BASE_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "common/registry.hpp"
#include "math/extmath.hpp"

namespace sgdlib {

class LossFunction {
protected:
    LossParamType loss_param_;
public:
    LossFunction(LossParamType loss_param): loss_param_(loss_param) {};
    virtual ~LossFunction() {};

    virtual double evaluate(const std::vector<FeatureType>& X, 
                            const std::vector<LabelType>& y, 
                            const std::vector<FeatureType>& weight, 
                            const FeatureType& bias) const = 0 ;

    virtual void gradient(const std::vector<FeatureType>& X, 
                          const std::vector<LabelType>& y, 
                          const std::vector<FeatureType>& weight,
                          const FeatureType& bias,
                          std::vector<FeatureType>& weight_grad, 
                          FeatureType& bias_grad) const = 0;
};

// Create registries for base loss function
DECLARE_REGISTRY(LossFunctionRegistry, LossFunction, LossParamType);
DEFINE_REGISTRY(LossFunctionRegistry, LossFunction, LossParamType);

} // namespace sgdlib

#endif // CORE_LOSS_BASE_HPP_