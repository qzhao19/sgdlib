#ifndef MATH_MATH_KERNELS_OPS_SSE_FLOAT_HPP_
#define MATH_MATH_KERNELS_OPS_SSE_FLOAT_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

/**
 * @brief Clips each element in the input array `x` to be within the range [min, max].
 *        using SSE intrinsics for float.
 *
 * This function uses SSE intrinsics to process 4 float elements at a time for performance.
 * Any remaining elements are processed individually using `std::clamp`.
 *
 * @param min The minimum value to clip to.
 * @param max The maximum value to clip to.
 * @param x A pointer to the input array of floating-point numbers.
 * @param n The number of elements in the array `x`.
 *
 */
inline void vec_clip_sse_float(float min, float max, float* x, std::size_t n) noexcept {
    // if n < 4
    if (n < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = std::clamp(x[i], min, max);
        }
        return;
    }

    // load const values min/max to SIMD register
    const __m128 xmin = _mm_set1_ps(min);
    const __m128 xmax = _mm_set1_ps(max);

    // compute aligned bound of array
    const std::size_t aligned_size = n & ~3ULL;
    const float* aligned_bound = x + aligned_size;

    // process the array in chunks of 4 elements
    for (float* ptr = x; ptr < aligned_bound; ptr += 4) {
        __m128 vec = _mm_loadu_ps(ptr);
        vec = _mm_min_ps(_mm_max_ps(vec, xmin), xmax);
        _mm_storeu_ps(ptr, vec);
    }


    for (std::size_t i = aligned_size; i < n; ++i) {
        x[i] = std::clamp(x[i], min, max);
    }
};

/**
 * @brief Checks if any element in the input array `x` is infinite (either +∞ or -∞)
 *        using SSE. intrinsics for float.
 *
 * This function uses SSE intrinsics to process 4 float elements at a time for performance.
 * Early returns when any infinite value is detected. Processes remaining elements with
 * std::isinf after the SIMD operations.
 *
 * @param x A pointer to the input array of floating-point numbers.
 * @param n The number of elements in the array `x`.
 * @return true If any element in the array is ±∞
 * @return false If all elements are finite
 */
inline bool isinf_sse_float(const float* x, std::size_t n) noexcept {
    if (n == 0) return false;
    if (n < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isinf(x[i])) {
                return true;
            }
        }
        return false;
    }

    // keep ptr to x
    const float* const xbegin = x;

    // load const val to register
    const __m128 pos_inf = _mm_set1_ps(std::numeric_limits<float>::infinity());
    const __m128 neg_inf = _mm_set1_ps(-std::numeric_limits<float>::infinity());

    // compute aligned bound = xsize - xsize % 4
    const float* ptr = x;
    const float* aligned_bound = x + (n & ~3ULL);

    // processed the array in chunks of 4 elems
    for (; ptr < aligned_bound; ptr += 4) {
        const __m128 vec = _mm_loadu_ps(ptr);
        const __m128 cmp = _mm_or_ps(_mm_cmpeq_ps(vec, pos_inf),
                                     _mm_cmpeq_ps(vec, neg_inf));

        if (_mm_movemask_ps(cmp) != 0) {
            return true;
        }
    }

    // process the rest of elems
    if (const std::size_t tail_size = n & 4ULL) {
        const float* tail_ptr = xbegin + (n & ~3ULL);
        for (std::size_t i = 0; i < tail_size; ++i) {
            if (std::isinf(tail_ptr[i])) {
                return true;
            }
        }
    }
    return false;
};



}
}
#endif // MATH_MATH_KERNELS_OPS_SSE_FLOAT_HPP_
