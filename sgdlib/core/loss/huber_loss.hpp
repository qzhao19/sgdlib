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
    HuberLoss(sgdlib::LossParamType loss_param): LossFunction(loss_param) {};
    ~HuberLoss() = default;

    sgdlib::FeatureScalarType evaluate(const sgdlib::FeatureScalarType &y_pred,
                                       const sgdlib::LabelScalarType &y_true) const override {
        return 0.0;
    }

    sgdlib::FeatureScalarType derivate(const sgdlib::FeatureScalarType &y_pred,
                                       const sgdlib::LabelScalarType &y_true) const override {
        return 0.0;
    }

    sgdlib::FeatureScalarType evaluate_with_gradient(const sgdlib::ArrayDatasetType &dataset,
                                                     const std::vector<sgdlib::FeatureScalarType> &w,
                                                     std::vector<sgdlib::FeatureScalarType> &grad) const override {
        return 0.0;
    }

};

} // namespace detail
} // namespace sgdlib

#endif // CORE_LOSS_HUBER_LOSS_HPP_
