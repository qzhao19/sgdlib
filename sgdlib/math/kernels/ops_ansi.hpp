#ifndef MATH_MATH_KERNELS_OPS_ANSI_HPP_
#define MATH_MATH_KERNELS_OPS_ANSI_HPP_

#include "common/logging.hpp"
#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

/**
 * @brief Clips the values in the input vector to be within the specified range.
 *
 * @param x The input vector to be clipped.
 * @param min The minimum value to clip to.
 * @param max The maximum value to clip to.
 *
 * @note This function modifies the input vector in place.
 */ 
template<typename T>
inline void clip_ansi(T min, T max, std::vector<T>& x) {
    std::transform(x.begin(), x.end(), x.begin(),
        [min, max](const T val) { return std::clamp(val, min, max); }
    );
};
    
}
}

#endif // MATH_MATH_KERNELS_OPS_ANSI_HPP_