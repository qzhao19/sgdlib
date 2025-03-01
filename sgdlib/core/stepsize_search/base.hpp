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
template <typename LossFuncType>
class StepSizeSearch {
protected:
    double lipschitz_;
    double linesearch_scaling_;
    std::shared_ptr<LossFuncType> loss_fn_;

    std::vector<FeatureType> X_;
    std::vector<LabelType> y_;
    StepSizeSearchParamType *stepsize_search_params_;
    
public:  
    StepSizeSearch(const std::vector<FeatureType>& X, 
                   const std::vector<LabelType>& y,
                   const std::shared_ptr<LossFuncType>& loss_fn,
                   StepSizeSearchParamType *stepsize_search_params): X_(X), y_(y),
                        loss_fn_(loss_fn),
                        stepsize_search_params_(stepsize_search_params){};
    
    ~StepSizeSearch() {};

    virtual int search(bool is_saga, double& step_size) {
        return 0;
    };

    virtual int search(const FeatureType& y_pred, 
                       const LabelType& y_true, 
                       const FeatureType& grad,
                       const FeatureType& xnorm, 
                       const std::size_t& step,
                       double& stepsize) {
        return 0;
    };

    virtual int search(const std::vector<FeatureType>& xp, 
                       const std::vector<FeatureType>& gp, 
                       std::vector<FeatureType>& x,
                       std::vector<FeatureType>& g, 
                       std::vector<FeatureType>& d,
                       FeatureType& fx,
                       double& stepsize) {
        return 0;
    };

};

} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_BASE_HPP_