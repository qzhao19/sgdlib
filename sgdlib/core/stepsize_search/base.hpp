#ifndef CORE_STEPSIZE_SEARCH_BASE_HPP_
#define CORE_STEPSIZE_SEARCH_BASE_HPP_

#include "common/constants.hpp"
#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "common/registry.hpp"
#include "common/params.hpp"
#include "common/types.hpp"
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
class StepSizeSearch {
protected:
    sgdlib::ScalarType lipschitz_;
    sgdlib::ScalarType linesearch_scaling_;
    sgdlib::ArrayDatasetType dataset_;
    std::shared_ptr<sgdlib::detail::LossFunctionType> loss_fn_;
    std::shared_ptr<sgdlib::StepSizeSearchParamType> stepsize_search_params_;

public:
    StepSizeSearch(const sgdlib::ArrayDatasetType &dataset,
                   const std::shared_ptr<sgdlib::detail::LossFunctionType>& loss_fn,
                   std::shared_ptr<sgdlib::StepSizeSearchParamType> stepsize_search_params):
                        dataset_(dataset),
                        loss_fn_(loss_fn),
                        stepsize_search_params_(stepsize_search_params){};

    virtual ~StepSizeSearch() = default;

    virtual int search(bool is_saga, sgdlib::ScalarType& step_size) {
        return 0;
    };

    virtual int search(const sgdlib::FeatureScalarType& y_pred,
                       const sgdlib::LabelScalarType& y_true,
                       const sgdlib::FeatureScalarType& grad,
                       const sgdlib::FeatureScalarType& xnorm,
                       const std::size_t& step,
                       sgdlib::ScalarType& stepsize) {
        return 0;
    };

    virtual int search(const std::vector<sgdlib::FeatureScalarType>& xp,
                       const std::vector<sgdlib::FeatureScalarType>& gp,
                       const std::vector<sgdlib::FeatureScalarType>& d,
                       std::vector<sgdlib::FeatureScalarType>& x,
                       std::vector<sgdlib::FeatureScalarType>& g,
                       sgdlib::FeatureScalarType& fx,
                       sgdlib::ScalarType& stepsize) {
        return 0;
    };

};

using StepSizeSearchType = StepSizeSearch;

} // namespace detail
} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_BASE_HPP_
