#ifndef MATH_MATH_KERNELS_OPS_SSE_FLOAT_HPP_
#define MATH_MATH_KERNELS_OPS_SSE_FLOAT_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

/**
 * Clips (clamps) elements of a float vector to specified [min, max] range using SSE intrinsics.
 * Performs in-place modification: x[i] = min(max(x[i], min), max)
 *
 * @param[in,out] x    Pointer to input/output vector (must be non-null and at least length n)
 * @param[in] min      Lower bound of clipping range (inclusive)
 * @param[in] max      Upper bound of clipping range (inclusive)
 * @param[in] n        Number of elements to process
 *
 * @note
 * - Uses SSE4.2 instruction set (requires CPU support)
 * - Processes 4 elements per cycle in main computation loop
 * - For small vectors (n < 4), falls back to scalar operation
 * - Behavior is undefined if min > max
 * - More efficient than std::transform with std::clamp for large vectors
 * - No explicit alignment requirements but performance improves with 16-byte aligned data
 *
 * @example
 *   float data[] = {-1.0f, 0.5f, 2.0f, 3.0f};
 *   vecclip_sse_float(data, 0.0f, 1.0f, 4);
 *   // data becomes [0.0f, 0.5f, 1.0f, 1.0f]
 *
 * @see _mm_min_ps, _mm_max_ps (Intel Intrinsics Guide)
 * @see std::clamp (alternative for scalar clipping)
 */
inline void vecclip_sse_float(float* x, float min, float max, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return ;
    if (min > max) return ;
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
 * Checks if a float vector contains any infinity values using SSE intrinsics.
 *
 * @param[in] x  Pointer to the input vector (must be non-null and at least length n)
 * @param[in] n  Number of elements in the vector
 *
 * @return true if any element is ±infinity, false otherwise
 *         Returns false if:
 *         - x is nullptr
 *         - n == 0
 *
 * @note
 * - Uses SSE4.2 instruction set (requires CPU support)
 * - Processes 4 elements per cycle in the main computation loop
 * - For small vectors (n < 4), falls back to scalar checking
 * - Detects both positive and negative infinity
 * - More efficient than std::any_of with std::isinf for large vectors
 *
 * @example
 *   float data[] = {1.0f, 2.0f, INFINITY, 4.0f};
 *   bool result = hasinf_sse_float(data, 4);  // Returns true
 *
 *   float clean[] = {1.0f, 2.0f, 3.0f};
 *   bool result2 = hasinf_sse_float(clean, 3); // Returns false
 *
 * @see _mm_cmpeq_ps, _mm_movemask_ps (Intel Intrinsics Guide)
 * @see std::isinf (alternative for scalar checking)
 */
inline bool hasinf_sse_float(const float* x, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return false;
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
 * Computes the L2 norm (Euclidean norm) or squared L2 norm of a float vector using SSE intrinsics.
 *
 * @param[in] x        Pointer to the input vector (must be 16-byte aligned for optimal performance)
 * @param[in] n        Number of elements in the vector
 * @param[in] squared  When true, returns the squared norm (avoiding sqrt computation)
 *
 * @return The L2 norm (if squared=false) or squared L2 norm (if squared=true) of the input vector.
 *         Returns 0.0f if:
 *         - x is nullptr
 *         - n == 0
 *
 * @note
 * - Uses SSE4.2 instruction set (requires CPU support)
 * - Processes 4 elements per cycle in the main computation loop
 * - For small vectors (n < 4), falls back to scalar computation
 * - The squared option provides faster computation when the exact norm isn't needed
 * - No explicit alignment requirements but performance improves with aligned data
 *
 * @example
 *   // Compute regular L2 norm
 *   float x[] = {3.0f, 4.0f, 0.0f, 0.0f};
 *   float norm = vecnorm2_sse_float(x, 4, false);  // Returns 5.0f (sqrt(3²+4²))
 *
 *   // Compute squared norm
 *   float squared_norm = vecnorm2_sse_float(x, 4, true);  // Returns 25.0f (3²+4²)
 *
 * @see _mm_mul_ps, _mm_sqrt_ps, _mm_hadd_ps (Intel Intrinsics Guide)
 * @see https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm
 */
inline float vecnorm2_sse_float(const float* x, std::size_t n, bool squared) noexcept {
    if (n == 0 || x == nullptr) return 0.0f;
    if (n < 4) {
        float sum = 0.0f;
        switch (n) {
            case 3: sum += x[2] * x[2]; [[fallthrough]];
            case 2: sum += x[1] * x[1]; [[fallthrough]];
            case 1: sum += x[0] * x[0];
            default: break;
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
        case 3: total += x[n - 3] * x[n - 3]; [[fallthrough]];
        case 2: total += x[n - 2] * x[n - 2]; [[fallthrough]];
        case 1: total += x[n - 1] * x[n - 1];
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
            default: break;
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
        case 3: total += std::abs(x[n - 3]); [[fallthrough]];
        case 2: total += std::abs(x[n - 2]); [[fallthrough]];
        case 1: total += std::abs(x[n - 1]);
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
 * @example:
 *   float data[100];
 *   vecscale_sse_float(data, 100, 2.5f); // data *= 2.5
 */
inline void vecscale_sse_float(const float* x,
                               const float c,
                               std::size_t n,
                               float* out) noexcept {
    // conditionn check
    if (x == nullptr || out == nullptr) return;
    if (c == 1.0f) return ;

    // for small size array
    if (n < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i] * c;
        }
        return;
    }

    // compute aligned bound na dptr points to x
    const float* ptr = x;
    const float* aligned_bound = x + (n & ~3ULL);

    // load constant c into register
    const __m128 scalar = _mm_set1_ps(c);

    // maon SIMD loop
    for (; ptr < aligned_bound; ptr += 4, out += 4) {
        const __m128 xvec = _mm_loadu_ps(ptr);
        // const __m128 outvec= _mm_mul_ps(xvec, scalar);
        _mm_storeu_ps(out, _mm_mul_ps(xvec, scalar));
    }

    // tail handling
    // const std::size_t remaining = n & 3;
    // const std::size_t offset = n - remaining;
    // switch (n & 3ULL) {
    //     case 3: out[n - 3] = x[n - 3] * c; [[fallthrough]];
    //     case 2: out[n - 2] = x[n - 2] * c; [[fallthrough]];
    //     case 1: out[n - 1] = x[n - 1] * c;
    //     default: break;
    // }
    const std::size_t remaining = n & 3;
    if (remaining != 0) {
        const std::size_t offset = n - remaining;
        for (std::size_t i = 0; i < remaining; ++i) {
            out[offset + i] = x[offset + i] * c;
        }
    }
};


/**
 * Computes the dot product of two single-precision floating-point vectors
 * using SSE (Streaming SIMD Extensions) intrinsics.
 *
 * @param[in] x Pointer to the first input vector
 * @param[in] y Pointer to the second input vector
 * @param[in] n Number of elements in vector x
 * @param[in] m Number of elements in vector y (must equal n for valid computation)
 *
 * @return The dot product of vectors x and y as a single-precision float.
 *         Returns 0.0f if:
 *         - Either x or y is nullptr
 *         - Vector lengths n and m don't match
 *         - n == 0
 *
 * @note
 * - Uses SSE4.1 instruction set (requires CPU support)
 * - Processes 4 elements per cycle in the main computation loop
 * - Automatically handles remaining elements (1-3) when n is not a multiple of 4
 * - For small vectors (n < 4), back to scalar computation
 * - No explicit alignment requirements but performance improves with aligned data
 * - Provides approximately 4x speedup over scalar implementation for large vectors
 *
 * @example
 *   float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
 *   float y[] = {5.0f, 6.0f, 7.0f, 8.0f};
 *   float result = vecdot_sse_float(x, y, 4, 4);  // Returns 70.0f (1*5 + 2*6 + 3*7 + 4*8)
 *
 * @see _mm_mul_ps, _mm_add_ps, _mm_hadd_ps (Intel Intrinsics Guide)
 */
inline float vecdot_sse_float(const float* x, const float* y, std::size_t n, std::size_t m) noexcept {
    if (x == nullptr || y == nullptr) return 0.0f;
    if (n != m) return 0.0f;
    if (n == 0) return 0.0f;

    // handle small size case n < 4
    if (n < 4 && m < 4) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    // get aligned array bound
    const float* xptr = x;
    const float* yptr = y;
    const float* aligned_bound = x + (n & ~3ULL);

    // load sum of vec to register
    __m128 sum = _mm_setzero_ps();

    // loop array x and y in chunks of 4 elems
    for (; xptr < aligned_bound; xptr += 4, yptr += 4) {
        const __m128 xvec = _mm_loadu_ps(xptr);
        const __m128 yvec = _mm_loadu_ps(yptr);
        sum = _mm_add_ps(sum, _mm_mul_ps(xvec, yvec));
    }

    // 1st impelmetation
    // const __m128 shuf = _mm_movehdup_ps(sum);
    // const __m128 sums = _mm_add_ps(shuf, sum);
    // const __m128 sumh = _mm_add_ps(sum, _mm_movehl_ps(sums, sums));

    // 2ed implementation
    // _mm_movehl_ps: [a,b,c,d] => [c,d,c,d]
    // _mm_add_ps: tmp = [a+c,b+d,c+c,d+d]
    // 0x55 = 0b01'01'01'01，
    // _mm_shuffle_ps(tmp, tmp, 0x55) → [b+d, b+d, b+d, b+d]
    const __m128 shuf = _mm_movehl_ps(sum, sum);
    const __m128 tmp = _mm_add_ps(sum, shuf);
    const __m128 sumh = _mm_add_ps(tmp, _mm_shuffle_ps(tmp, tmp, 0x55));
    float total = _mm_cvtss_f32(sumh);

    switch (n & 3ULL) {
        case 3: total += x[n - 3] * y[n - 3]; [[fallthrough]];
        case 2: total += x[n - 2] * y[n - 2]; [[fallthrough]];
        case 1: total += x[n - 1] * y[n - 1];
        default: break;
    }
    return total;
};


inline void vecscale_sse_float(const float* xbegin,
                            const float* xend,
                            const float c,
                            std::size_t n,
                            float* out) noexcept {
    if (xbegin == nullptr || xend == nullptr ||
        xend <= xbegin || out == nullptr) {
        return;
    }
    if (n == 0 || c == 1.0f) return ;

    const std::size_t m = static_cast<std::size_t>(xend - xbegin);
    if (n != m) return ;

    // for small size n < 4
    if (n < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = xbegin[i] * c;
        }
        return ;
    }
    // define ptr to xbegin and aligned end
    const float* ptr = xbegin;
    const float* aligned_bound = xbegin + (n & ~3ULL);

    // load scalar to register
    const __m128 scalar = _mm_set1_ps(c);

    // main SIMD processing, primary vectorized loop
    for (; ptr < aligned_bound; ptr += 4, out += 4) {
        const __m128 xvec = _mm_loadu_ps(ptr);
        const __m128 outvec = _mm_mul_ps(xvec, scalar);
        _mm_storeu_ps(out, outvec);
    }

    // tail handling
    const std::size_t offset = n - (n & 3ULL);
    switch (n & 3ULL) {
        case 3: out[offset + 2] = xbegin[offset + 2] * c; [[fallthrough]];
        case 2: out[offset + 1] = xbegin[offset + 1] * c; [[fallthrough]];
        case 1: out[offset] = xbegin[offset] * c;
        default: break;
    }
}


}
}
#endif // MATH_MATH_KERNELS_OPS_SSE_FLOAT_HPP_
