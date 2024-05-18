#ifndef COMMON_MATH_HPP_
#define COMMON_MATH_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {
namespace internal {

/**
 * @brief compute sigmoid function 
 *      f(x) = 1/(1 + e^-x)
*/
template <typename Type>
Type sigmoid(Type x) {
    return 1 / (1 + std::exp(-x));
};

} // namespace internal
} // namespace sgdlib

#endif // COMMON_MATH_HPP_