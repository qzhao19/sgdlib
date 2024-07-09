#ifndef CORE_STEPSIZE_SEARCH_BASE_HPP_
#define CORE_STEPSIZE_SEARCH_BASE_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "common/registry.hpp"
#include "core/loss.hpp"
#include "math/extmath.hpp"

namespace sgdlib {

/** 
 * @file base.hpp
 * 
 * @class StepSizeSearch
 * 
 * @brief Abstract base class representing step size search
 * 
*/
class StepSizeSearch {
protected:
    std::vector<FeatureType> X_;
    std::vector<LabelType> y_;
    StepSizeSearchParamType stepsize_search_param_;
    
public:
    StepSizeSearch(const std::vector<FeatureType>& X, 
                   const std::vector<LabelType>& y,
                   StepSizeSearchParamType stepsize_search_param): X_(X), y_(y),
                stepsize_search_param_(stepsize_search_param) {};
    ~StepSizeSearch() {};

    virtual int search(bool is_saga, double& step_size) {
        return 0;
    };
};

} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_BASE_HPP_