#ifndef MATH_EXTMATH_HPP_
#define MATH_EXTMATH_HPP_

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

#endif // MATH_EXTMATH_HPP_