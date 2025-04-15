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
inline void vecclip_ansi(T min, T max, std::vector<T>& x) {
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
inline bool hasinf_ansi(const std::vector<T>& x) {
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
inline T vecnorm2_ansi(const std::vector<T>& x, bool squared) {
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
inline T vecnorm1_ansi(const std::vector<T>& x) {
    return std::accumulate(x.begin(), x.end(), static_cast<T>(0),
        [](T acc, const T& value) {
            return acc + std::abs(value);
        });
};

/**
 * @brief Applies a scalar multiplication operation to a vector.
 *
 * @tparam T The type of elements in the vector.
 *
 * @param[in,out] x vector of type T, which will be scaled by the scalar 'c'.
 * @param[in] c constant scalar value
 *
 * @note The function is marked as inline, which is suitable for small functions
 *       to reduce the overhead of function calls.
*/
template<typename T>
inline void vecscale_ansi(std::vector<T>& x, const T& c) {
    std::transform(x.begin(), x.end(),
                   x.begin(),
                  [&c](const T& val) { return val * c; });
}




}
}
#endif // MATH_MATH_KERNELS_OPS_ANSI_HPP_
