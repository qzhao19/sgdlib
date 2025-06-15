#ifndef COMMON_TYPES_HPP_
#define COMMON_TYPES_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {

#ifndef PRECISION_TYPE
#define PRECISION_TYPE   64
#endif /*PRECISION_TYPE*/

// base value type
#if     PRECISION_TYPE == 32
using ScalarType = float;
#elif   PRECISION_TYPE == 64
using ScalarType = double;
#else
#error "sgdlib supports single (PRECISION_TYPE=32) and double (PRECISION_TYPE=64) precision floating-point formats exclusively."
#endif

using FeatureType = ScalarType;
using LabelType = int;

using LossParamType = std::unordered_map<std::string, ScalarType>;
using LRDecayParamType = std::unordered_map<std::string, ScalarType>;

// forward declaration StepSizeSearchParam
namespace detail {
    struct StepSizeSearchParam;
}
using StepSizeSearchParamType = sgdlib::detail::StepSizeSearchParam;

}

#endif
