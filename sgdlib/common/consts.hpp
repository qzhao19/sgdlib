#ifndef COMMON_CONSTS_HPP_
#define COMMON_CONSTS_HPP_

#include "common/prereqs.hpp"

#ifndef PRECISION_TYPE
#define PRECISION_TYPE   64
#endif/*PRECISION_TYPE*/

// base value type
#if     PRECISION_TYPE == 32
using FloatType = float;
using IntegerType = int;
#elif   PRECISION_TYPE == 64
using FloatType = double;
using IntegerType = long;
#else
#error "sgdlib supports (float; PRECISION_TYPE = 32) or FloatType (FloatType; PRECISION_TYPE=64) precision only."
#endif

// define the type for feature, label, loss parameters and learning rate decay parameters
using FeatValType = FloatType;
using LabelValType = IntegerType;
using LossParamType = std::unordered_map<std::string, FloatType>;
using LRDecayParamType = std::unordered_map<std::string, FloatType>;

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

struct StepSizeSearchParam {
    // regularization coefficients for L2
    FloatType alpha;

    // initial learning rate
    FloatType eta0;

    // Controls the rate of step reduction,
    // decreasing the step size until the condition is met.
    FloatType dec_factor;

    // increase coefficient, Control the increase rate of the step size,
    // used to enlarge the step size when the step size is too small,
    // to avoid the convergence problem caused by the step size is too small.
    FloatType inc_factor;

    // parameter to control the accuracy of the line search
    FloatType ftol;

    // coefficient for the Wolfe condition,which
    // is valid only when the backtracking line-search
    FloatType wolfe;

    // maximum step of the line search routine
    FloatType max_stepsize;

    // minimum step of the line search routine
    FloatType min_stepsize;

    // maximum number of iterations
    std::size_t max_iters;

    // maximum number of trials for the line search
    std::size_t max_searches;

    // Armijo condition or wolfe condition
    std::string condition;
};

using StepSizeSearchParamType = StepSizeSearchParam;
static StepSizeSearchParamType DEFAULT_STEPSIZE_SEARCH_PARAMS = {
    0.0, 0.01, 0.5, 2.1,
    1e-4, 0.9, 1e+20, 1e-20,
    20, 10, "WOLFE"
};

constexpr FloatType max_dloss = 1e+10;
#define MAX_DLOSS max_dloss

constexpr FloatType wscale_threshold = 1e-9;
#define WSCALE_THRESHOLD wscale_threshold

constexpr FloatType inf = std::numeric_limits<FloatType>::infinity();
#define INF inf

#ifndef M_PI
    #define M_PI 3.141592653589793238462643383279
#endif

const std::unordered_map<std::string, std::string> LOSS_FUNCTION = {{"log_loss", "LogLoss"}};
const std::unordered_map<std::string, std::string> LEARNING_RATE = {{"invscaling", "Invscaling"},
                                                                    {"exponential", "Exponential"}};
const std::unordered_set<std::string> PENALTY_TYPES = {"none", "l1", "l2"};

#endif // COMMON_CONSTS_HPP_
