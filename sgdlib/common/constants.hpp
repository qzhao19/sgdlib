#ifndef COMMON_CONSTS_HPP_
#define COMMON_CONSTS_HPP_

#include "common/prereqs.hpp"
#include "common/types.hpp"

namespace sgdlib {
namespace detail {

enum class TaskType {
    Classification,
    Regression
};

enum {
    // unknown error
    LBFGS_ERROR_UNKNOWNERROR = -1024,

    // insufficient memory
    LBFGS_ERROR_OUTOFMEMORY,

    // invalid stopping criterion
    LBFGS_ERROR_INVALID_PARAMETERS,

    // increase gradient
    LBFGS_ERROR_INCREASE_GRADIENT,

    // line-search step became smaller than min_stepsize
    LBFGS_ERROR_MINIMUM_STEPSIZE,

    // line-search step became larger than min_stepsize
    LBFGS_ERROR_MAXIMUM_STEPSIZE,

    // line-search routine reaches the maximum number of evaluations
    LBFGS_ERROR_MAXIMUM_SEARCHES,
};

constexpr sgdlib::ScalarType MAX_DLOSS = 1e+10;
constexpr sgdlib::ScalarType WSCALE_THRESHOLD = 1e-9;
constexpr sgdlib::ScalarType INF = std::numeric_limits<sgdlib::ScalarType>::infinity();

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279
#endif

}
}
#endif // COMMON_CONSTS_HPP_
