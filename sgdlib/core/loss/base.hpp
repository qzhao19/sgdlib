#ifndef CORE_LOSS_BASE_HPP_
#define CORE_LOSS_BASE_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"

namespace sgdlib {

class LossFunction {
public:
    LossFunction() {};
    virtual ~LossFunction() {};

    virtual double evaluate(const std::vector<FeatureType>& X, 
                            const std::vector<LabelType>& y, 
                            const std::vector<FeatureType>& w) const = 0 ;

    virtual void gradient(const std::vector<FeatureType>& X, 
                          const std::vector<LabelType>& y, 
                          const std::vector<FeatureType>& w,
                          std::vector<FeatureType>& grad) const = 0;
};

} // namespace sgdlib

#endif // CORE_LOSS_BASE_HPP_