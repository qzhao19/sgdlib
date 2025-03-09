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
    FloatType lipschitz_;
    FloatType linesearch_scaling_;
    std::shared_ptr<LossFuncType> loss_fn_;

    std::vector<FeatValType> X_;
    std::vector<LabelValType> y_;
    StepSizeSearchParamType *stepsize_search_params_;
    
public:  
    StepSizeSearch(const std::vector<FeatValType>& X, 
                   const std::vector<LabelValType>& y,
                   const std::shared_ptr<LossFuncType>& loss_fn,
                   StepSizeSearchParamType *stepsize_search_params): X_(X), y_(y),
                        loss_fn_(loss_fn),
                        stepsize_search_params_(stepsize_search_params){};
    
    virtual ~StepSizeSearch() = default;

    virtual int search(bool is_saga, FloatType& step_size) {
        return 0;
    };

    virtual int search(const FeatValType& y_pred, 
                       const LabelValType& y_true, 
                       const FeatValType& grad,
                       const FeatValType& xnorm, 
                       const std::size_t& step,
                       FloatType& stepsize) {
        return 0;
    };

    virtual int search(const std::vector<FeatValType>& xp, 
                       const std::vector<FeatValType>& gp, 
                       std::vector<FeatValType>& x,
                       std::vector<FeatValType>& g, 
                       std::vector<FeatValType>& d,
                       FeatValType& fx,
                       FloatType& stepsize) {
        return 0;
    };

};

} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_BASE_HPP_