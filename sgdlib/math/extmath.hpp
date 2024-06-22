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
};

/**
 * @brief check if any element of vector is infinite.
 * 
 * @param x vector containing elements to check infinity.
*/
template<typename Type>
bool isinf(const std::vector<Type>& x) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (std::isinf(x[i])) {
            return true;
        }
    }
    return false;
}

/**
 * @brief check if the given value represents infinity
 * 
 * @param x value to check infinity.
 */
template<typename Type>
bool isinf(const Type& x) {
    if (std::isinf(x)) {
        return true;
    }
    return false;
}

/**
 * @brief calculate the L2 norm of a vector.
 * 
 * @param x a vector of type T.
 * @return The L2 norm of the vector as a double.
 */
template<typename Type>
double sqnorm2(const std::vector<Type>& x) {
    return std::sqrt(
        std::inner_product(x.begin(), x.end(), x.begin(), 0.0)
    );
}

/**
 * @brief Computes the index of the maximum element in a given array.
 *
 * @param x Pointer to the first element of the array to be processed.
 *         The array should contain 'size' number of elements of type ValueType.
 * @param size The number of elements in the array pointed to by 'x'.
 * @return IndexType The index of the maximum element within the array.
 *         If multiple elements are equal to the maximum value, the index of the first occurrence is returned.
 */
template<typename ValueType, typename IndexType>
IndexType argmax(ValueType* x, unsigned long size) {
    IndexType max_index = 0;
    ValueType max_value = x[max_index];

    for (unsigned long i = 0; i < size; i++) {
        if (x[i] > max_value) {
            max_index = i;
            max_value = x[max_index];
        }
    }
    return max_index;
};

} // namespace internal
} // namespace sgdlib

#endif // MATH_EXTMATH_HPP_