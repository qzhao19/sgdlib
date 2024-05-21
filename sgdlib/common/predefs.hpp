#ifndef COMMON_PREDEFS_HPP_
#define COMMON_PREDEFS_HPP_

#include "common/prereqs.hpp"

using FeatureType = double;
using LabelType = long;

constexpr double max_dloss = 1e+20;
constexpr double min_dloss = 1e-20;
#define MAX_DLOSS max_dloss;
#define MIN_DLOSS min_dloss;

#ifndef M_PI
#define M_PI 3.14159265358979323846
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


#endif // COMMON_PREDEFS_HPP_
