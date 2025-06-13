#ifndef CORE_STEPSIZE_SEARCH_BASE_HPP_
#define CORE_STEPSIZE_SEARCH_BASE_HPP_

#include "common/constants.hpp"
#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "common/registry.hpp"
#include "core/loss.hpp"
#include "math/math_ops.hpp"

namespace sgdlib {
namespace detail {

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
    sgdlib::detail::ArrayDatasetType dataset_;
    std::shared_ptr<StepSizeSearchParamType> stepsize_search_params_;

public:
    StepSizeSearch(const sgdlib::detail::ArrayDatasetType &dataset,
                   const std::shared_ptr<LossFuncType>& loss_fn,
                   std::shared_ptr<StepSizeSearchParamType> stepsize_search_params):
                        dataset_(dataset),
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
                       const std::vector<FeatValType>& d,
                       std::vector<FeatValType>& x,
                       std::vector<FeatValType>& g,
                       FeatValType& fx,
                       FloatType& stepsize) {
        return 0;
    };

};

} // namespace detail
} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_BASE_HPP_
