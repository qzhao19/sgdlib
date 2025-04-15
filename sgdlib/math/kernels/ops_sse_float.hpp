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
inline void vecclip_sse_float(float* x, float min, float max, std::size_t n) noexcept {
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

    if (aligned_size) {
        for (std::size_t i = aligned_size; i < n; ++i) {
            x[i] = std::clamp(x[i], min, max);
        }
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
inline bool hasinf_sse_float(const float* x, std::size_t n) noexcept {
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
    return (n & 3ULL) ? std::isinf(x[n - 1])
        || (n & 2ULL ? std::isinf(x[n - 2]) : false)
        || (n & 1ULL ? std::isinf(x[n - 3]) : false) :
        false;
};

/**
 * @brief Computes the L2 norm of the input array `x` using SSE intrinsics for acceleration.
 *
 * This function leverages SSE intrinsics to process 4 floating-point elements at a time for
 * improved performance. For small arrays with fewer than 4 elements or the remaining elements
 * after SIMD processing, it performs scalar computations. It can return either the squared
 * L2 norm or its square root based on the `squared` parameter.
 *
 * @param x A pointer to the input array of floating-point numbers.
 * @param n The number of elements in the array `x`.
 * @param squared If `true`, returns the squared L2 norm; if `false`, returns the square root of the L2 norm.
 * @return float The computed L2 norm (either squared or its square root depending on the `squared` parameter).
 */
inline float vecnorm2_sse_float(const float* x, std::size_t n, bool squared) noexcept {
    if (n == 0) return 0.0f;
    if (n < 4) {
        float sum = 0.0f;
        switch (n) {
            case 3: sum += x[2] * x[2]; [[fallthrough]];
            case 2: sum += x[1] * x[1]; [[fallthrough]];
            case 1: sum += x[0] * x[0];
        }
        return squared ? sum : std::sqrt(sum);
    }

    // compute aligned bound = xsize - xsize % 4
    const float* ptr = x;
    const float* aligned_bound = x + (n & ~3ULL);

    __m128 sum = _mm_setzero_ps();
    for (; ptr < aligned_bound; ptr += 4) {
        const __m128 vec = _mm_loadu_ps(ptr);
        sum = _mm_add_ps(sum, _mm_mul_ps(vec, vec));
    }

    const __m128 shuf = _mm_movehdup_ps(sum);  // [a,b,c,d] -> [b,b,d,d]
    const __m128 sums = _mm_add_ps(sum, shuf); // [a+b, b+b, c+d, d+d]
    const __m128 sumv = _mm_add_ss(sums, _mm_movehl_ps(sums, sums)); // add [a+b, c+d] -> [a+b+c+d]
    float total = _mm_cvtss_f32(sumv);

    switch (n & 3ULL) {
        case 3: total += x[n - 1] * x[n - 1]; [[fallthrough]];
        case 2: total += x[n - 2] * x[n - 2]; [[fallthrough]];
        case 1: total += x[n - 3] * x[n - 3];
        default: break;
    }

    return squared ? total : std::sqrt(total);
}

/**
 * Computes the L1 norm (sum of absolute values) of a float array using SSE intrinsics.
 *
 * @param x     Pointer to the input float array (must not be nullptr unless n=0)
 * @param n     Number of elements in the array
 * @return      The L1 norm as a float value
 *
 * @note This function uses SSE4.1 intrinsics for vectorized computation:
 *       - Processes 4 elements per iteration when n >= 4
 *       - Handles remaining elements (1-3) separately
 *       - Properly manages NaN values (preserves them in the sum)
 *       - Zero overhead when n=0 or x=nullptr
 *
 * @compiler Requires SSE4.2 support (-msse4.2 flag)
 * @exception noexcept guaranteed
 * @complexity O(n) with 4x speedup for large arrays
 *
 * Example:
 *   float data[8] = {...};
 *   float norm = vecnorm1_sse_float(data, 8); // sum(|data[i]|)
 */
inline float vecnorm1_sse_float(const float* x, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return 0.0f;
    if (n < 4) {
        float sum = 0.0f;
        switch (n) {
            case 3: sum += std::abs(x[2]); [[fallthrough]];
            case 2: sum += std::abs(x[1]); [[fallthrough]];
            case 1: sum += std::abs(x[0]);
        }
        return sum;
    }

    // compute aligned bound = xsize - xsize % 4
    const float* ptr = x;
    const float* aligned_bound = x + (n & ~3ULL);

    // _mm_set1_epi32: set all elements to 0x7FFFFFFF,
    // contains 4 32bit int value, each value is 0x7FFFFFFF
    // _mm_castsi128_ps: convert int to float
    const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));

    __m128 sum = _mm_setzero_ps();
    for (; ptr < aligned_bound; ptr += 4) {
        const __m128 vec = _mm_loadu_ps(ptr);
        // _mm_and_ps: bitwise '&' operation between mask and vec
        sum = _mm_add_ps(sum, _mm_and_ps(vec, abs_mask));
    }

    // sum = [a,b,c,d] -> [a,b,c,d] -> [a+b, b+c, c+d, d+0] -> [a+b+c+d, c+d, d+0, 0]
    const __m128 shuf = _mm_movehdup_ps(sum);  // [a,b,c,d] -> [b,b,d,d]
    const __m128 sums = _mm_add_ps(sum, shuf); // [a+b, b+b, c+d, d+d]
    const __m128 sumv = _mm_add_ss(sums, _mm_movehl_ps(sums, sums)); // add [a+b, c+d] -> [a+b+c+d]
    float total = _mm_cvtss_f32(sumv);

    switch (n & 3ULL) {
        case 3: total += std::abs(x[n - 1]); [[fallthrough]];
        case 2: total += std::abs(x[n - 2]); [[fallthrough]];
        case 1: total += std::abs(x[n - 3]);
        default: break;
    }
    return total;
};

/**
 * Scales a float array in-place by constant factor using SSE vectorization.
 *
 * Performs element-wise multiplication: x[i] = x[i] * c for all i in [0,n-1]
 *
 * @param[in,out] x  Pointer to float array to scale (modified in-place)
 * @param[in]     n  Number of elements in array
 * @param[in]     c  Scaling factor
 *
 * @note Optimized implementation details:
 *       - Uses SSE SIMD to process 4 elements per cycle
 *       - Handles unaligned memory accesses safely
 *       - Special cases:
 *           - Returns immediately if x == nullptr c == 1.0f
 *       - Processes remaining 1-3 elements after SIMD loop
 *
 * @requirements SSE4.2 instruction set support (-msse4.2)
 * @exception noexcept guaranteed
 * @complexity O(n) with ~4x speedup vs scalar
 *
 * Example:
 *   float data[100];
 *   vecscale_sse_float(data, 100, 2.5f); // data *= 2.5
 */
inline void vecscale_sse_float(float* x, std::size_t n, const float c) noexcept {
    if (x == nullptr || c == 1.0f) return;

    // for small size array
    if (n < 4) {
        switch (n) {
            case 3: x[2] *= c; [[fallthrough]];
            case 2: x[1] *= c; [[fallthrough]];
            case 1: x[0] *= c;
        }
        return;
    }

    // compute bound of array x and aligned bound
    const float* bound = x + n;
    const float* aligned_bound = x + (n & ~3ULL);

    // load scalar to register
    const __m128 scalar = _mm_set1_ps(c);

    // _mm_mul_ps: x[i] * c
    for (; x < aligned_bound; x += 4) {
        __m128 vec = _mm_mul_ps(_mm_loadu_ps(x), scalar);
        _mm_storeu_ps(x, vec);
    }

    // handle remaining elements
    switch (bound - x){
        case 3: x[2] *= c; [[fallthrough]];
        case 2: x[1] *= c; [[fallthrough]];
        case 1: x[0] *= c;
        default: break;
    }
};


}
}
#endif // MATH_MATH_KERNELS_OPS_SSE_FLOAT_HPP_
