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

    sgdlib::FeatureType evaluate(const sgdlib::FeatureType &y_pred,
                                 const sgdlib::LabelType &y_true) const override {
        return 0.0;
    }

    sgdlib::FeatureType derivate(const sgdlib::FeatureType &y_pred,
                                 const sgdlib::LabelType &y_true) const override {
        return 0.0;
    }

    sgdlib::FeatureType evaluate_with_gradient(const sgdlib::ArrayDatasetType &dataset,
                                               const std::vector<sgdlib::FeatureType> &w,
                                               std::vector<sgdlib::FeatureType> &grad) const override {
        return 0.0;
    }

};

} // namespace detail
} // namespace sgdlib

#endif // CORE_LOSS_HUBER_LOSS_HPP_
