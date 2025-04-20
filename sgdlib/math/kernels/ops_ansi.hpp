#ifndef MATH_MATH_KERNELS_OPS_ANSI_HPP_
#define MATH_MATH_KERNELS_OPS_ANSI_HPP_

#include "common/logging.hpp"
#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

/**
 * @brief Clips the values in the input vector to be within the specified range.
 * @tparam T The type of elements in the vector.
 * @param x The input vector to be clipped.
 * @param min The minimum value to clip to.
 * @param max The maximum value to clip to.
 *
 * @note This function modifies the input vector in place.
 */
template<typename T>
inline void vecclip_ansi(std::vector<T>& x, T min, T max) noexcept {
    std::transform(x.begin(), x.end(), x.begin(),
        [min, max](const T val) {
            return std::clamp(val, min, max);
        }
    );
};

/**
 * @brief Checks if any element in the input vector is infinite.
 *
 * @tparam T The type of elements in the vector.
 *
 * @param x The input vector to be checked.
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
 * @param x a vector of type T.
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
 * @param x a vector of type T.
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
 * @brief Computes element-wise scaling: out[i] = x[i] * c
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
}

/**
 * @brief Applies a scalar multiplication operation to a vector.
 * It computes the sum of the products of all elements within
 * the iterator range from begin to end with a constant c.
 *
 * @tparam T The type of elements in the vector.
 *
 * @param[in] x vector of type T,
 * @param[in] y vector of type T,
 * @param out The reference to a vector that stores the result of the dot product.
 *
 * @note The function is marked as inline, which is suitable for small functions
 *       to reduce the overhead of function calls.
 *
 */
template<typename T>
inline T vecdot_ansi(const std::vector<T>& x, const std::vector<T>& y) noexcept {
    T prod = std::inner_product(x.begin(), x.end(),
                                y.begin(),
                                static_cast<T>(0));
    return prod;
}

/**
 * Computes out[i] = x[i] * c for all elements in [xbegin, xend)
 *
 * @tparam T Numeric type (float, double, int, etc.)
 *
 * @param xbegin Pointer to the first input element
 * @param xend Pointer to one past the last input element
 * @param c Scalar multiplier
 * @param out Output vector (will be resized to match input size)
 *
 * @note
 * - Provides strong exception safety
 * - Prevents aliasing issues
 */
template<typename T>
inline void vecadd_ansi(const T* xbegin,
                        const T* xend,
                        const T c,
                        std::vector<T>& out) {
    // more safe
    out.assign(xbegin, xend);
    std::transform(out.begin(), out.end(), out.begin(),
                    [c](T val) { return val * c; });
};



}
}
#endif // MATH_MATH_KERNELS_OPS_ANSI_HPP_
