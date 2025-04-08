#ifndef MATH_MATH_KERNELS_OPS_SSE41_FLOAT_HPP_
#define MATH_MATH_KERNELS_OPS_SSE41_FLOAT_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

/**
 * @brief Clips each element in the input array `x` to be within the range [min, max].
 *        using SSE4.1 intrinsics for float.
 * 
 * This function uses SSE4.1 intrinsics to process 4 float elements at a time for performance.
 * Any remaining elements are processed individually using `std::clamp`.
 * 
 * @param min The minimum value to clip to.
 * @param max The maximum value to clip to.
 * @param x A pointer to the input array of floating-point numbers.
 * @param n The number of elements in the array `x`.
 * 
 */
inline void clip_sse41_float(float min, float max, float* x, std::size_t n) {
    // if n < 4
    if (n < 4) {
        for (size_t i = 0; i < n; ++i) {
            x[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
        }
        return;
    }

    // Create an SSE register with all elements set to the min/max value
    const __m128 xmin = _mm_set1_ps(min);
    const __m128 xmax = _mm_set1_ps(max);
    // Initialize the loop index
    size_t i = 0;

    // Process the array in chunks of 4 elements using SSE4.1 intrinsics
    // The loop will run until i is less than n - 3
    // The loop will increment i by 4 in each iteration
    for (; i + 3 < n; i += 4) {
        __m128 vec = _mm_loadu_ps(x + i);
        vec = _mm_min_ps(_mm_max_ps(vec, xmin), xmax);
        _mm_storeu_ps(x + i, vec);
    }

    // Process any remaining elements using std::clamp
    for (; i < n; ++i) {
        x[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    }
};

}
}

#endif // MATH_MATH_KERNELS_OPS_SSE_FLOAT_HPP_