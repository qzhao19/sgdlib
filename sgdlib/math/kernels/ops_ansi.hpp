#ifndef MATH_MATH_KERNELS_OPS_ANSI_HPP_
#define MATH_MATH_KERNELS_OPS_ANSI_HPP_

#include "common/logging.hpp"
#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {


template<typename T>
void vecset(const T c, std::vector<T>& x) {
    std::fill(x.begin(), x.end(), c);
}

template<typename T>
void veccpy(const std::vector<T>& x, std::vector<T>& out) {
    std::copy(x.begin(), x.end(), out.begin());
}

template<typename T>
void vecncpy(const std::vector<T>& x, std::vector<T>& out) {
    std::transform(x.begin(), x.end(),
                   out.begin(),
                   [](const T& val) { return -val; });
}

/**
 * @brief Clips the values in the input vector to be within the specified range.
 *
 * @tparam T The type of elements in the vector.
 *
 * @param[in, out] x The input vector to be clipped.
 * @param[in] min The minimum value to clip to.
 * @param[in] max The maximum value to clip to.
 *
 * @note This function modifies the input vector in place.
 */
template<typename T>
inline void vecclip_ansi(std::vector<T>& x, T min, T max) noexcept {
    std::transform(x.begin(), x.end(), x.begin(),
        [min, max](const T& val) {
            return std::clamp(val, min, max);
        }
    );
};

/**
 * @brief Checks if any element in the input vector is infinite.
 *
 * @tparam T The type of elements in the vector.
 *
 * @param[in] x The input vector to be checked.
 *
 * @return True if any element in the input vector is infinite, false otherwise.
 */
template<typename T>
inline bool hasinf_ansi(const std::vector<T>& x) noexcept {
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (std::isinf(x[i])) {
            return true;
        }
    }
    return false;
};

/**
 * @brief calculate the L2 norm of a vector.
 *
 * @tparam T The type of elements in the vector.
 *
 * @param[in] x a vector of type T.
 * @return The L2 norm of the vector as a custome type.
 *
 * @note The function is marked as inline, which is suitable for small functions
 *       to reduce the overhead of function calls.
 */
template<typename T>
inline T vecnorm2_ansi(const std::vector<T>& x, bool squared) noexcept {
    if (x.empty()) return 0.0;
    T l2_norm = std::inner_product(x.begin(), x.end(),
                                   x.begin(),
                                   static_cast<T>(0));
    return squared ? l2_norm : std::sqrt(l2_norm);
};

/**
 * @brief calculate the L1 norm (Manhattan distance) of a vector.
 *
 * @tparam T The type of elements in the vector.
 *
 * @param[in] x a vector of type T.
 * @return The L1 norm of the vector as a T.
 */
template<typename T>
inline T vecnorm1_ansi(const std::vector<T>& x) noexcept {
    return std::accumulate(x.begin(), x.end(), static_cast<T>(0),
        [](T acc, const T& value) {
            return acc + std::abs(value);
        });
};

/**
 * @brief Computes element-wise scale multiplication : out[i] = x[i] * c
 *
 * @tparam T Numeric type (float, double, int, etc.)
 *
 * @param[in] x     Input vector
 * @param[in] c     Scalar multiplier
 * @param[out] out  Output vector (will be resized to match x)
 *
 * @note The function is marked as inline, which is suitable for small functions
 *       to reduce the overhead of function calls.
 */
template<typename T>
inline void vecscale_ansi(const std::vector<T>& x,
                          const T& c,
                          std::vector<T>& out) noexcept {
    std::transform(x.begin(), x.end(),
                   out.begin(),
                   [&c](const T& val) { return val * c; });
};

/**
 * @brief Computes element-wise scale multiplication out[i] = x[i] * c
 *        for all elements in [xbegin, xend)
 *
 * @tparam T Numeric type (float, double, int, etc.)
 *
 * @param[in] xbegin Pointer to the first input element
 * @param[in] xend Pointer to one past the last input element
 * @param[in] c Scalar multiplier
 * @param[out] out Output vector (will be resized to match input size)
 *
 * @note
 * - Provides strong exception safety
 * - Prevents aliasing issues
 */
template<typename T>
inline void vecscale_ansi(const T* xbegin,
                          const T* xend,
                          const T& c,
                          std::vector<T>& out) noexcept {
    // more safe
    out.assign(xbegin, xend);
    std::transform(out.begin(), out.end(),
                   out.begin(),
                   [&c](const T& val) { return val * c; });
};

/**
 * Adds two vectors element-wise and stores the result in an output vector.
 *
 * @tparam T The data type of the elements in the vectors.
 *
 * @param[in] x The first input vector.
 * @param[in] y The second input vector.
 * @param[out] out The output vector where the result of the element-wise addition will be stored.
 *
 * @example
 * std::vector<int> vec1 = {1, 2, 3};
 * std::vector<int> vec2 = {4, 5, 6};
 * std::vector<int> result(3);
 * vecadd_ansi(vec1, vec2, result);
 * // result will be {5, 7, 9}
 */
template<typename T>
inline void vecadd_ansi(const std::vector<T>& x,
                        const std::vector<T>& y,
                        std::vector<T>& out) noexcept {
    out.resize(x.size());
    std::transform(x.begin(), x.end(),
                   y.begin(),
                   out.begin(),
                   std::plus<T>());
};

/**
 * @brief Computes the element-wise scaled addition of two vectors.
 *
 * This function performs an element-wise operation on two input vectors `x` and `y`,
 * where each element of the result is calculated as \( x_{\text{val}} \times c + y_{\text{val}} \).
 *
 * @tparam T The data type of the elements in the vectors.
 *
 * @param x The first input vector.
 * @param y The second input vector.
 * @param c The constant factor to scale the elements of the first input vector `x`.
 * @param out The output vector where the result of the element-wise scaled addition will be stored.
 *
 * @example
 * std::vector<int> vec1 = {1, 2, 3};
 * std::vector<int> vec2 = {4, 5, 6};
 * int scale_factor = 2;
 * std::vector<int> result(3);
 * vecadd_ansi(vec1, vec2, scale_factor, result);
 * // result will be {6, 9, 12} because (1*2 + 4)=6, (2*2 + 5)=9, (3*2 + 6)=12
 *
 */
template<typename T>
inline void vecadd_ansi(const std::vector<T>& x,
                        const std::vector<T>& y,
                        const T& c,
                        std::vector<T>& out) noexcept {
    out.resize(x.size());
    std::transform(x.begin(), x.end(),
                   y.begin(),
                   out.begin(),
                   [&c](const T& xval, const T& yval) {
                        return xval * c + yval;
                   });
};

/**
 * @brief compute the element-wise difference between two vectors.
 *
 * @tparam T The data type of the elements in the vectors.
 *
 * @param x The first input vector.
 * @param y The second input vector.
 * @param out The output vector where the result of the element-wise difference will be stored.
 *
 * @example
 * std::vector<int> vec1 = {1, 2, 3};
 * std::vector<int> vec2 = {4, 5, 6};
 * std::vector<int> result(3);
 * vecdiff_ansi(vec1, vec2, result);
 * // result will be {-3, -3, -3}
 */
template<typename T>
inline void vecdiff_ansi(const std::vector<T>& x,
                        const std::vector<T>& y,
                        std::vector<T>& out) noexcept {
    out.resize(x.size());
    std::transform(x.begin(), x.end(),
                   y.begin(),
                   out.begin(),
                   std::minus<T>());
};

/**
 * @brief Applies a scalar multiplication operation to a vector.
 * It computes the sum of the products of all elements within
 * the iterator range from begin to end with a constant c.
 *
 * @tparam T The type of elements in the vector.
 *
 * @param[in] x vector of type T,
 * @param[in] y vector of type T,
 * @param[out] out The reference to a vector that stores the result of the dot product.
 *
 * @note The function is marked as inline, which is suitable for small functions
 *       to reduce the overhead of function calls.
 *
 */
template<typename T>
inline T vecdot_ansi(const std::vector<T>& x,
                     const std::vector<T>& y) noexcept {
    T prod = std::inner_product(x.begin(), x.end(),
                                y.begin(),
                                static_cast<T>(0));
    return prod;
};

/**
 * @brief Element-wise multiplication of two vectors.
 *
 * This function performs element-wise multiplication of two input
 * vectors `x` and `y` and stores the result in the `out` vector.
 *
 * @tparam T The data type of the elements in the vectors.
 *
 * @param[in]  x   The first input vector.
 * @param[in]  y   The second input vector.
 * @param[out] out The output vector where the result of the
 *                 element-wise multiplication will be stored.
 *
 * @example
 * std::vector<int> vec1 = {1, 2, 3};
 * std::vector<int> vec2 = {4, 5, 6};
 * std::vector<int> result(3);
 * vecmul_ansi(vec1, vec2, result);
 * // result will be {4, 10, 18}
 */
template<typename T>
inline void vecmul_ansi(const std::vector<T>& x,
                        const std::vector<T>& y,
                        std::vector<T>& out) noexcept {
    out.resize(x.size());
    std::transform(x.begin(), x.end(),
                   y.begin(),
                   out.begin(),
                   std::multiplies<T>());
};


}
}
#endif // MATH_MATH_KERNELS_OPS_ANSI_HPP_
