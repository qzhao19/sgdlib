#ifndef CORE_LOSS_BASE_HPP_
#define CORE_LOSS_BASE_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "common/registry.hpp"
#include "math/extmath.hpp"

namespace sgdlib {

/** 
 * @file base.hpp
 * 
 * @class LossFunction
 * 
 * @brief Abstract base class representing a loss function
 * 
*/
class LossFunction {
protected:
    LossParamType loss_param_;
public:
    LossFunction(LossParamType loss_param): loss_param_(loss_param) {};
    virtual ~LossFunction() {};

    /**
     * evaluate the loss value of loss function
     * 
     * @param y_pred FeatureType
     *      The predicted output of a model given an input.
     * @param y_true LabelType
     *      The true label value for the data point
     * @return A double value for loss function value
    */
    virtual FeatureType evaluate(const FeatureType& y_pred, 
                                 const LabelType& y_true) const = 0;
    
    /** 
     * compute the derivative of a loss function.
     * 
     * @param y_pred FeatureType
     *      The predicted output of a model given an input.
     * @param y_true LabelType
     *      The true label value for the data point
     * @return A double value representing the derivative of the loss function
    */
    virtual FeatureType derivate(const FeatureType& y_pred, 
                                 const LabelType& y_true) const = 0;

    /**
     * @brief Sets a parameter for the loss function.
     * 
     * @param name The name of the parameter as a string.
     * @param value The value of the parameter as a double.
    */
    void set_param(const std::string& name, const double& value) {
        loss_param_[name] = value;
    }
};

// Create registries for base loss function
DECLARE_REGISTRY(LossFunctionRegistry, LossFunction, LossParamType);
DEFINE_REGISTRY(LossFunctionRegistry, LossFunction, LossParamType);

} // namespace sgdlib

#endif // CORE_LOSS_BASE_HPP_