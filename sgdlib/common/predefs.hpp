#ifndef COMMON_PREDEFS_HPP_
#define COMMON_PREDEFS_HPP_

#include "common/prereqs.hpp"

using FeatureType = double;
using LabelType = long;
using LossParamType = std::unordered_map<std::string, double>;
using LRDecayParamType = std::unordered_map<std::string, double>;
using StepSizeSearchParamType = std::unordered_map<std::string, double>;

constexpr double max_dloss = 1e+10;
#define MAX_DLOSS max_dloss

constexpr double wscale_threshold = 1e-9;
#define WSCALE_THRESHOLD wscale_threshold

constexpr double inf = std::numeric_limits<double>::infinity();
#define INF inf

#ifndef M_PI
    #define M_PI 3.141592653589793238462643383279
#endif

// disable copy and assign a class 
#ifndef DISABLE_COPY_AND_ASSIGN
#define DISABLE_COPY_AND_ASSIGN(classname)           \
    classname(const classname&) = delete;            \
    classname& operator=(const classname&) = delete
#endif


// declare anonymous variable
#define CONCATENATE_IMPL(var1, var2)  var1##var2
#define CONCATENATE(var1, var2) CONCATENATE_IMPL(var1, var2)
#define ANONYMOUS_VARIABLE(name) CONCATENATE(name, __LINE__)

#define UNUSED __attribute__((__unused__))
#define USED __attribute__((__used__))

std::string Demangle(const char* name) {
    int status;
    auto demangled = ::abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (demangled) {
        std::string ret;
        ret.assign(demangled);
        free(demangled);
        return ret;
    }
    return name;
}

template <typename Type>
static const char* DemangleType() {
#ifdef __GXX_RTTI
    static const std::string name = Demangle(typeid(Type).name());
    return name.c_str();
#else
    return "(RTTI disabled, cannot show name)";
#endif
}

const std::unordered_map<std::string, std::string> LOSS_FUNCTION = {{"log_loss", "LogLoss"}};
const std::unordered_map<std::string, std::string> LEARNING_RATE = {{"invscaling", "Invscaling"}, 
                                                                    {"exponential", "Exponential"}};
const std::unordered_set<std::string> PENALTY_TYPES = {"none", "l1", "l2"};

#endif // COMMON_PREDEFS_HPP_
