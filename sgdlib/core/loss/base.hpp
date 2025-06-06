#ifndef CORE_LOSS_BASE_HPP_
#define CORE_LOSS_BASE_HPP_

#include "common/constants.hpp"
#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "common/registry.hpp"
#include "math/math_ops.hpp"

namespace sgdlib {
namespace detail {

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
    virtual ~LossFunction() = default;

    /**
     * evaluate the loss value of loss function
     *
     * @param y_pred FeatValType
     *      The predicted output of a model given an input.
     * @param y_true LabelValType
     *      The true label value for the data point
     * @return A FloatType value for loss function value
    */
    virtual FeatValType evaluate(const FeatValType& y_pred,
                                 const LabelValType& y_true) const = 0;

    /**
     * compute the derivative of a loss function.
     *
     * @param y_pred FeatValType
     *      The predicted output of a model given an input.
     * @param y_true LabelValType
     *      The true label value for the data point
     * @return A FloatType value representing the derivative of the loss function
    */
    virtual FeatValType derivate(const FeatValType& y_pred,
                                 const LabelValType& y_true) const = 0;


    /**
     * Computes both loss value and gradient vector in a single pass
     *
     * @param X Feature matrix stored as a flattened 1D array (row-major order)
     * @param y Label vector corresponding to the feature matrix
     * @param w Model parameter vector to evaluate
     * @param[out] grad Gradient vector (must be pre-allocated with correct size)
     *
     * @return FeatValType Computed loss value
     */
    virtual FeatValType evaluate_with_gradient(
        const std::vector<FeatValType>& X,
        const std::vector<LabelValType>& y,
        const std::vector<FeatValType>& w,
        std::vector<FeatValType>& grad) const = 0;

    /**
     * @brief Sets a parameter for the loss function.
     *
     * @param name The name of the parameter as a string.
     * @param value The value of the parameter as a FloatType.
    */
    void set_param(const std::string& name, const FloatType& value) {
        loss_param_[name] = value;
    }
};

// Create registries for base loss function
DECLARE_SHARED_REGISTRY(LossFunctionRegistry, LossFunction, LossParamType);
DEFINE_SHARED_REGISTRY(LossFunctionRegistry, LossFunction, LossParamType);

} // namespace detail
} // namespace sgdlib

#endif // CORE_LOSS_BASE_HPP_
