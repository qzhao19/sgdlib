#ifndef COMMON_PARAMS_HPP_
#define COMMON_PARAMS_HPP_

#include "common/prereqs.hpp"
#include "common/types.hpp"

namespace sgdlib {
namespace detail {

struct StepSizeSearchParam {
    // regularization coefficients for L2
    sgdlib::ScalarType alpha;

    // initial learning rate
    sgdlib::ScalarType eta0;

    // Controls the rate of step reduction,
    // decreasing the step size until the condition is met.
    sgdlib::ScalarType dec_factor;

    // increase coefficient, Control the increase rate of the step size,
    // used to enlarge the step size when the step size is too small,
    // to avoid the convergence problem caused by the step size is too small.
    sgdlib::ScalarType inc_factor;

    // parameter to control the accuracy of the line search
    sgdlib::ScalarType ftol;

    // coefficient for the Wolfe condition,which
    // is valid only when the backtracking line-search
    sgdlib::ScalarType wolfe;

    // maximum step of the line search routine
    sgdlib::ScalarType max_stepsize;

    // minimum step of the line search routine
    sgdlib::ScalarType min_stepsize;

    // maximum number of iterations
    std::size_t max_iters;

    // maximum number of trials for the line search
    std::size_t max_searches;

    // Armijo condition or wolfe condition
    std::string condition;
};

inline StepSizeSearchParamType DEFAULT_STEPSIZE_SEARCH_PARAMS = {
    0.0, 0.01, 0.5, 2.1,
    1e-4, 0.9, 1e+20, 1e-20,
    20, 10, "WOLFE"
};

inline const std::unordered_map<std::string, std::string> LOSS_FUNCTION = {
    {"log_loss", "LogLoss"}
};

inline const std::unordered_map<std::string, std::string> LEARNING_RATE = {
    {"invscaling", "Invscaling"},
    {"exponential", "Exponential"}
};

inline const std::unordered_set<std::string> PENALTY_TYPES = {
    "none", "l1", "l2"
};

}
}

#endif // COMMON_PARAMS_HPP_
