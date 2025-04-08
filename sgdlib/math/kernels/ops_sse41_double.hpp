#ifndef MATH_MATH_KERNELS_OPS_SSE41_DOUBLE_HPP_
#define MATH_MATH_KERNELS_OPS_SSE41_DOUBLE_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

/**
 * @brief Clips each element in the input array `x` to be within the range [min, max] 
 *        using SSE4.1 intrinsics for doubles.
 * 
 * This function first processes the array in chunks of 2 double elements using SSE4.1 
 * intrinsics for performance. Any remaining elements are processed individually 
 * using a ternary operator to clamp them.
 * 
 * @param min The minimum value to clip to.
 * @param max The maximum value to clip to.
 * @param x A pointer to the input array of double-precision floating-point numbers.
 * @param n The number of elements in the array `x`.

 */
inline void clip_sse41_double(double min, double max, double* x, std::size_t n) {
    // Create an SSE register with all elements set to the min/max value
    const __m128d xmin = _mm_set1_pd(min);
    const __m128d xmax = _mm_set1_pd(max);
    const __m128d xnon = _mm_set1_pd(NAN);
    // Initialize the loop index
    std::size_t i = 0;

    // Process the array in chunks of 2 elements using SSE4.1 intrinsics
    for (; i + 1 < n; i += 2) {
        __m128d vec = _mm_loadu_pd(x + i);

        // check nan value of vec, nan = 1, otherwise is 0
        __m128d nan_mask = _mm_cmpunord_pd(vec, vec);

        // clamp the vector to the range [min, max]
        __m128d clipped = _mm_min_pd(_mm_max_pd(vec, xmin), xmax);

        // _mm_and_pd: keep nan val of vec, nan = 1, otherwise is 0
        // _mm_andnot_pd: keep non-nan val of clipped vec, nan = 0, otherwise is 1
        // _mm_or_pd: combine the two results
        vec = _mm_or_pd(_mm_and_pd(nan_mask, vec), 
                        _mm_andnot_pd(nan_mask, clipped));
        _mm_storeu_pd(x + i, vec);
    }

    // Process any remaining elements
    if (i < n) {
        x[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    }
}


}
}
#endif // MATH_MATH_KERNELS_OPS_SSE_FLOAT_HPP_