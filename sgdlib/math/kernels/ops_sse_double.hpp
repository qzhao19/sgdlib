#ifndef MATH_MATH_KERNELS_OPS_SSE_DOUBLE_HPP_
#define MATH_MATH_KERNELS_OPS_SSE_DOUBLE_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

/**
 * @brief Clips each element in the input array `x` to be within the range [min, max]
 *        using SSE intrinsics for doubles.
 *
 * This function first processes the array in chunks of 2 double elements using SSE
 * intrinsics for performance. Any remaining elements are processed individually
 * using a ternary operator to clamp them.
 *
 * @param min The minimum value to clip to.
 * @param max The maximum value to clip to.
 * @param x A pointer to the input array of double-precision floating-point numbers.
 * @param n The number of elements in the array `x`.

 */
inline void vec_clip_sse_double(double min, double max, double* x, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return ;
    // Create an SSE register with all elements set to the min/max value
    const __m128d xmin = _mm_set1_pd(min);
    const __m128d xmax = _mm_set1_pd(max);

    // check if x is aligned to 16 bytes
    if (reinterpret_cast<uintptr_t>(x) % 16 != 0) {
        x[0] = std::clamp(x[0], min, max);
        x++;
        n--;
        if (n == 0) return ;
    }

    // compute aligned bound
    std::size_t aligned_size = n & ~1ULL;
    double* aligned_bound = x + aligned_size;

    // Process the array in chunks of 2 elements using SSE intrinsics
    for (double* ptr = x; ptr < aligned_bound; ptr += 2) {
        __m128d vec = _mm_loadu_pd(ptr);

        // clamp the vector to the range [min, max]
        __m128d clipped = _mm_min_pd(_mm_max_pd(vec, xmin), xmax);

        // nan_mask: check nan value of vec, nan = 1, otherwise is 0
        // _mm_and_pd: keep nan val of vec, nan = 1, otherwise is 0
        // _mm_andnot_pd: keep non-nan val of clipped vec, nan = 0, otherwise is 1
        // _mm_or_pd: combine the two results
        __m128d nan_mask = _mm_cmpunord_pd(vec, vec);
        vec = _mm_or_pd(_mm_and_pd(nan_mask, vec),
                        _mm_andnot_pd(nan_mask, clipped));
        _mm_storeu_pd(ptr, vec);
    }

    // Process any remaining elements
    if (aligned_size < n) {
        x[aligned_size] = std::clamp(x[aligned_size], min, max);
    }
}

/**
 * @brief Checks if any element in the input array `x` is infinite (either +∞ or -∞)
 *        using SSE intrinsics for double-precision floats.
 *
 * This function processes 2 elements at a time using SSE intrinsics, with early exit
 * when infinity is detected. Handles remaining elements with scalar operations.
 *
 * @param x Pointer to the array of double-precision floating-point numbers
 * @param n Number of elements in the array
 * @return true If any element is ±∞
 * @return false If all elements are finite or array is empty
 */
inline bool vec_isinf_sse_double(const double* x, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return false;

    // check if x is aligned to 16 bytes
    if (reinterpret_cast<uintptr_t>(x) % 16 != 0) {
        if (std::isinf(x[0])) {
            return true;
        }
        x++;
        n--;
        if (n == 0) return false;
    }

    // load x positif inf + x negative inf to SIMD register
    const __m128d pos_inf = _mm_set1_pd(std::numeric_limits<double>::infinity());
    const __m128d neg_inf = _mm_set1_pd(-std::numeric_limits<double>::infinity());

    // compute aligned bound
    const double* ptr = x;
    const double* aligned_bound = x + (n & ~1ULL);

    // loop the array in chunks of 2 elements
    for (; ptr < aligned_bound; ptr += 2) {
        const __m128d vec = _mm_loadu_pd(ptr);
        const __m128d cmp = _mm_or_pd(_mm_cmpeq_pd(vec, pos_inf),
                                      _mm_cmpeq_pd(vec, neg_inf));
        if (_mm_movemask_pd(cmp) != 0) {
            return true;
        }
    }

    // process remaining elements
    return (n & 1ULL) ? std::isinf(x[n - 1]) : false;
}

/**
 * @brief Computes the L2 norm of the input array `x` using SSE intrinsics for
 *        accelerated double-precision floating-point calculations.
 *
 * This function utilizes SSE intrinsics to process two double-precision
 * floating-point elements at a time to enhance performance. For arrays with
 * a small number of elements (less than or equal to two) or remaining elements
 * after SIMD processing, it performs scalar computations. It can return either
 * the squared L2 norm or its square root based on the `squared` parameter.
 *
 * @param x A pointer to the input array of double-precision floating-point numbers.
 * @param n The number of elements in the array `x`.
 * @param squared If `true`, returns the squared L2 norm; if `false`, returns the square root of the L2 norm.
 * @return double The computed L2 norm (either squared or its square root depending on the `squared` parameter).
 */
inline double vec_norm2_sse_double(const double* x, std::size_t n, bool squared) noexcept {
    if (n == 0 || x == nullptr) return 0.0;
    // For small arrays with less than or equal to two elements
    if (n <= 2) {
        double sum = 0.0;
        switch (n) {
            case 2: sum += x[1] * x[1]; [[fallthrough]];
            case 1: sum += x[0] * x[0];
        }
        return squared ? sum : std::sqrt(sum);
    }

    // compute aligned boundary
    const double* ptr = x;
    const double* aligned_bound = x + (n & ~1ULL);

    // init sum to 0
    __m128d sum = _mm_setzero_pd();
    for (; ptr < aligned_bound; ptr += 2) {
        // load 2 elements from x to SIMD register
        const __m128d vec = _mm_loadu_pd(ptr);
        // compute sum = sum + vec * vec
        sum = _mm_add_pd(sum, _mm_mul_pd(vec, vec));
    }

    double total;
    // perform a horizontal addition of the two channels' values in the SSE register
    __m128d sumh = _mm_hadd_pd(sum, sum);
    _mm_store_sd(&total, sumh);

    // process remaining elements
    if (n & 1ULL) {
        total += x[n - 1] * x[n - 1];
    }

    return squared ? total : std::sqrt(total);
}



}
}
#endif // MATH_MATH_KERNELS_OPS_SSE_FLOAT_HPP_
