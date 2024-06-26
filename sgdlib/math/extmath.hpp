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
inline Type sigmoid(Type x) {
    return 1 / (1 + std::exp(-x));
};

/**
 * @brief Clip (limit) the values in a vector.
 * 
 * @param x vector containing elements to clip.
 * @param min, max minimum and maximum value
*/
template<typename Type>
inline void clip(std::vector<Type>& x, Type min, Type max) {
    if (min > max) {
        throw std::invalid_argument("a_min must be less than or equal to a_max.");
    }
    std::transform(std::begin(x), std::end(x), std::begin(x),
        [=] (auto i) { 
            return std::clamp(i, min, max); 
        }
    );
};

/** 
 * @brief Clips the value to the specified range [min, max]
 * 
 * @param x The input value to be clipped.
 * @param min, max minimum and maximum value
*/
template<typename Type>
inline void clip(Type& x, Type min, Type max) {
    if (min > max) {
        throw std::invalid_argument("a_min must be less than or equal to a_max.");
    }
    x = std::max(min, std::min(x, max));
}

/**
 * @brief check if any element of vector is infinite.
 * 
 * @param x vector containing elements to check infinity.
*/
template<typename Type>
inline bool isinf(const std::vector<Type>& x) {
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
inline bool isinf(const Type& x) {
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
inline double sqnorm2(const std::vector<Type>& x) {
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
inline IndexType argmax(ValueType* x, unsigned long size) {
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

/** 
 * @brief Applies a scalar multiplication operation to a vector.
 * 
 * @param[in,out] x vector of type T, which will be scaled by the scalar 'c'.
 * @param[in] scalar constant scalar value 
*/
template<typename T>
inline void dot(std::vector<T>& x, const T c) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] *= c;
    }
}

} // namespace internal
} // namespace sgdlib

#endif // MATH_EXTMATH_HPP_