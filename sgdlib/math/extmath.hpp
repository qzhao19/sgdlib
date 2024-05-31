#ifndef MATH_EXTMATH_HPP_
#define MATH_EXTMATH_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {
namespace internal {

/**
 * @brief compute sigmoid function 
 *      f(x) = 1/(1 + e^-x)
 * @param x input value
*/
template <typename Type>
Type sigmoid(Type x) {
    return 1 / (1 + std::exp(-x));
};

/**
 * @brief Clip (limit) the values in a vector.
 * 
 * @param x vector containing elements to clip.
 * @param min, max minimum and maximum value
*/
template<typename Type>
void clip(std::vector<Type>& x, Type min, Type max) {
    std::transform(std::begin(x), std::end(x), std::begin(x),
        [=] (auto i) { 
            return std::clamp(i, min, max); 
        }
    );
}

} // namespace internal
} // namespace sgdlib

#endif // MATH_EXTMATH_HPP_