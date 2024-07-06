#ifndef MATH_EXTMATH_HPP_
#define MATH_EXTMATH_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {
namespace internal {

/**
 * @brief Applies the sigmoid function to a given value.
 *
 * The sigmoid function is a mathematical function that can take any real-valued 
 * number and map it into a range between 0 and 1. It is often used in machine 
 * learning for converting a linear output into a probability-like value.
 *
 * The formula for the sigmoid function is:
 * \[ \text{sigmoid}(x) = \frac{1}{1 + e^{-x}} \]
 *
 * @tparam Type A floating-point type (e.g., float or double) for the input value.
 * @param x The input value to which the sigmoid function will be applied.
 * @return The result of applying the sigmoid function to the input value.
 *
 * @note This function assumes that the input value `x` is of a type that can be used in
 *       exponential calculations (usually a floating-point type).
*/
template <typename Type>
inline Type sigmoid(Type x) {
    return 1 / (1 + std::exp(-x));
};

/**
 * @brief Clip (limit) the values in a vector.
 * 
 * @tparam Type The type of elements in the vector.
 * 
 * @param x vector containing elements to clip.
 * @param min, max minimum and maximum value
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
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
 * @tparam Type The type of elements for inputs.
 * 
 * @param x The input value to be clipped.
 * @param min, max minimum and maximum value
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
*/
template<typename Type>
inline void clip(Type& x, Type min, Type max) {
    if (min > max) {
        throw std::invalid_argument("a_min must be less than or equal to a_max.");
    }
    x = std::max(min, std::min(x, max));
};

/**
 * @brief check if any element of vector is infinite.
 * 
 * @tparam Type The type of elements in the vector.
 * 
 * @param x vector containing elements to check infinity.
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
*/
template<typename Type>
inline bool isinf(const std::vector<Type>& x) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (std::isinf(x[i])) {
            return true;
        }
    }
    return false;
};

/**
 * @brief check if the given value represents infinity
 * 
 * @tparam Type The type of elements for input.
 * 
 * @param x value to check infinity.
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
 */
template<typename Type>
inline bool isinf(const Type& x) {
    if (std::isinf(x)) {
        return true;
    }
    return false;
};

/**
 * @brief calculate the L2 norm of a vector.
 * 
 * @tparam Type The type of elements in the vector.
 * 
 * @param x a vector of type T.
 * @return The L2 norm of the vector as a double.
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
 */
template<typename Type>
inline double sqnorm2(const std::vector<Type>& x) {
    return std::sqrt(
        std::inner_product(x.begin(), x.end(), x.begin(), 0.0)
    );
};

/**
 * @brief Computes the index of the maximum element in a given array.
 *
 * @param x Pointer to the first element of the array to be processed.
 *         The array should contain 'size' number of elements of type ValueType.
 * @param size The number of elements in the array pointed to by 'x'.
 * @return IndexType The index of the maximum element within the array.
 *         If multiple elements are equal to the maximum value, the index 
 *         of the first occurrence is returned.
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
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
 * @tparam Type The type of elements in the vector.
 * 
 * @param[in,out] x vector of type T, which will be scaled by the scalar 'c'.
 * @param[in] scalar constant scalar value 
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
*/
template<typename Type>
inline void dot(std::vector<Type>& x, const Type c) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] *= c;
    }
};

/** 
 * @brief Applies a scalar multiplication operation to a vector. 
 * It computes the sum of the products of all elements within 
 * the iterator range from begin to end with a constant c.
 * 
 * @tparam Type The type of elements in the vector.
 * @tparam IterType The type of iterator, with a default of std::vector<Type>::const_iterator.
 * 
 * @param begin The beginning iterator pointing to the first element 
 *      of the vector to calculate the dot product.
 * @param end The ending iterator pointing to the first element following begin.
 * @param c The constant to be multiplied with the elements of the vector.
 * @param out The reference to a vector that stores the result of the dot product.
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
 * @note The use of templates and default iterator types allows the function 
 *       to be used with different types of vectors and iterators.
 * 
*/
template<typename Type, 
         typename IterType = typename std::vector<Type>::const_iterator>
inline void dot(IterType begin, IterType end, 
                const Type c, 
                std::vector<Type>& out) {
    std::transform(begin, end, 
                   out.begin(), 
                   [c](const Type& elem) {
                        return elem * c;
                   });
};

/**
 * @brief Multiplies two vectors element-wise and stores the result in an output vector.
 *
 * This function takes two vectors of the same size, pointed to by v1 and v2, and their
 * element-wise products are calculated. The results are then stored in the output vector
 * pointed to by the out parameter.
 *
 * @tparam Type The data type of the elements in the input and output vectors.
 * @param v1 A constant reference to the first input vector.
 * @param v2 A constant reference to the second input vector. It must be the same size as v1.
 * @param out A reference to the output vector where the results will be stored. 
 *
 * @note This function assumes that v1 and v2 are of the same size. If they are not, the
 *       behavior is undefined. It is the caller's responsibility to ensure the sizes match.
 */
template<typename Type>
inline void multiply(const std::vector<Type>& v1, 
                     const std::vector<Type>& v2, 
                     std::vector<Type>& out) {
    std::transform(v1.begin(), v1.end(), v2.begin(), out.begin(),
                   [](const Type& a, const Type& b) { 
                        return a * b; 
                    });
}

} // namespace internal
} // namespace sgdlib

#endif // MATH_EXTMATH_HPP_