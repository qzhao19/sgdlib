#ifndef MATH_MATH_KERNELS_OPS_SSE_OPS_SSE_FLOAT_HPP_
#define MATH_MATH_KERNELS_OPS_SSE_OPS_SSE_FLOAT_HPP_

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
    // handle small size n < 4
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

    // handle small size n < 4
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
    // handle small size n < 4
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
 * @param[in]     x  Pointer to float array to scale (modified in-place)
 * @param[in]     n  Number of elements in array
 * @param[in]     c  Scaling factor
 * @param[out]    out Output vector (same size with x)
 *
 * @details Optimized implementation details:
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
    if (n == 0 || c == 1.0) return ;

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

    // handle remaining elements
    // const float* tail_ptr = x + (n & ~3ULL);
    switch (n & 3ULL) {
        case 3: out[2] = ptr[2] * c; [[fallthrough]];
        case 2: out[1] = ptr[1] * c; [[fallthrough]];
        case 1: out[0] = ptr[0] * c;
        default: break;
    }
};

/**
 * @brief Performs SIMD-accelerated vector scaling (element-wise multiplication) for
 *        single-precision floating-point arrays.
 *
 * Computes out[i] = x[i] * c for each element in the range [xbegin, xend) using SSE intrinsics.
 * Optimized for 32-bit floating-point data with automatic fallback to scalar operations.
 *
 * @param xbegin  [in] Pointer to the first element of the input array (must be valid if n > 0)
 * @param xend    [in] Pointer to one past the last element of the input array
 * @param c       [in] Scaling factor to multiply each element by
 * @param n       [in] Number of elements to process (must equal distance between xbegin and xend)
 * @param out     [out] Pointer to the output array (must have capacity for at least n elements)
 *
 * @note Implementation Details:
 * - Uses SSE instructions to process 4 floats simultaneously when n >= 4
 * - Automatically falls back to scalar operations for small arrays (n < 4)
 * - Handles remaining elements (n % 4) after SIMD processing
 * - No-throw guarantee (noexcept qualified)
 * - Memory-safe: validates pointers before access
 * - Supports unaligned memory accesses
 *
 * @warning Undefined behavior if:
 * - xbegin > xend (invalid range)
 * - n doesn't match actual array size
 * - Input and output ranges overlap
 * - Any pointer is null when n > 0
 * - Memory is not properly allocated
 *
 * @performance Expected to be 3-4x faster than scalar implementation for large arrays
 *
 * @example Basic Usage:
 * std::vector<float> data(1000, 2.0f);
 * std::vector<float> result(1000);
 * vecscale_sse_float(data.data(), data.data() + data.size(), 3.14f, data.size(), result.data());
 */
inline void vecscale_sse_float(const float* xbegin,
                               const float* xend,
                               const float c,
                               std::size_t n,
                               float* out) noexcept {
    if (xbegin == nullptr || xend == nullptr || out == nullptr) return;
    if (n == 0 || c == 1.0) return ;
    if (xend <= xbegin) return ;
    const std::size_t m = static_cast<std::size_t>(xend - xbegin);
    if (n != m) return ;

    // call vecscale_sse_float function
    vecscale_sse_float(xbegin, c, n, out);
};

/**
 * @brief Performs SIMD-accelerated element-wise addition of two float arrays.
 *
 * Computes out[i] = x[i] + y[i] for each element using SSE instructions when possible.
 * This function is optimized for float arrays and handles both aligned and unaligned memory.
 *
 * @param[in] x        Pointer to first input array (must not be nullptr unless n=0)
 * @param[in] y        Pointer to second input array (must not be nullptr unless n=0)
 * @param[in] n        Number of elements to process (primary size parameter)
 * @param[in] m        Secondary size parameter (unused in current implementation)
 * @param[out] out     Pointer to output array (must not be nullptr unless n=0)
 *
 * @details Safety checks:
 * - Returns immediately if x, y or out are nullptr (unless n=0)
 * - Returns if n=0 or m=0
 * - The m parameter is currently unused but kept for interface compatibility
 * - Uses SSE instructions (__m128) when n >= 4
 * - Falls back to scalar operations for small arrays (n < 4)
 * - Handles remaining elements (n % 4) after SIMD processing
 *
 * @example
 * - float vec1[] = {1.0f, 2.0f, 3.0f, 4.0f};
 * - float vec2[] = {5.0f, 6.0f, 7.0f, 8.0f};
 * - float result[4];
 * - vecadd_sse_float(vec1, vec2, 4, 4, result);
 * result = {6.0f, 8.0f, 10.0f, 12.0f}
 *
 * @note The function is marked as `noexcept` to indicate that it does not throw exceptions.
 *
 * @warning All pointers must point to valid memory if n > 0
 * @warning Behavior is undefined if input/output arrays overlap
 * @warning The m parameter is currently not used (m != m check is a no-op)
 */
inline void vecadd_sse_float(const float* x,
                             const float* y,
                             const std::size_t n,
                             const std::size_t m,
                             float* out) noexcept {
    if (x == nullptr || y == nullptr) return ;
    if (out == nullptr) return ;
    if (n == 0 || m == 0) return ;
    if (m != n) return ;

    // handle small size array n < 4
    if (n < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i] + y[i];
        }
        return ;
    }

    // define ptr points to x
    const float* xptr = x;
    const float* yptr = y;
    const float* aligned_bound = x + (n & ~3ULL);

    // main SIMD loop
    for (; xptr < aligned_bound; xptr += 4, yptr += 4, out += 4) {
        const __m128 xvec = _mm_loadu_ps(xptr);
        const __m128 yvec = _mm_loadu_ps(yptr);
        _mm_storeu_ps(out, _mm_add_ps(xvec, yvec));
    }

    // handle remaining elements
    switch (n & 3ULL) {
        case 3: out[2] = xptr[2] + yptr[2]; [[fallthrough]];
        case 2: out[1] = xptr[1] + yptr[1]; [[fallthrough]];
        case 1: out[0] = xptr[0] + yptr[0];
        default: break;
    }
};

/**
 * @brief Performs SIMD-accelerated vector addition with scaling for
 *        single-precision floating-point arrays.
 *
 * Computes out[i] = x[i] * c + y[i] for each element using SSE4.2 intrinsics.
 * Optimized for 32-bit floating-point data with automatic fallback to scalar operations.
 *
 * @param x       [in] Pointer to first input array (must be valid if n > 0)
 * @param y       [in] Pointer to second input array (must be valid if n > 0)
 * @param c       [in] Scaling factor to multiply x elements by before addition
 * @param n       [in] Primary size parameter (number of elements to process)
 * @param m       [in] Secondary size parameter (currently unused, reserved for future use)
 * @param out     [out] Pointer to output array (must have capacity for at least n elements)
 *
 * @note Implementation Details:
 * - Uses SSE instructions to process 4 floats simultaneously when n >= 4
 * - Automatically falls back to scalar operations for small arrays (n < 4)
 * - Handles remaining elements (n % 4) after SIMD processing
 * - No-throw guarantee (noexcept qualified)
 * - Supports unaligned memory accesses
 * - Current implementation ignores parameter m (reserved for future extension)
 *
 * @warning Undefined behavior if:
 * - Any pointer is null when n > 0
 * - Array sizes don't match (x, y, and out must all have at least n elements)
 * - Input and output ranges overlap
 * - n doesn't match actual array sizes
 *
 * @performance Expected to be 3-4x faster than scalar implementation for large arrays
 *
 * @example Basic Usage:
 * std::vector<float> x(1000, 2.0f);
 * std::vector<float> y(1000, 1.0f);
 * std::vector<float> result(1000);
 * vecadd_sse_float(x.data(), y.data(), 3.14f, x.size(), 0, result.data());
 */
inline void vecadd_sse_float(const float* x,
                             const float* y,
                             const float c,
                             const std::size_t n,
                             const std::size_t m,
                             float* out) noexcept {
    if (x == nullptr || y == nullptr) return ;
    if (out == nullptr) return ;
    if (n == 0 || m == 0) return ;
    if (m != n) return ;

    if (n < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = c * x[i] + y[i];
        }
        return ;
    }

    // define xptr and yptr point to x and y
    const float* xptr = x;
    const float* yptr = y;
    const float* aligned_bound = x + (n & ~3ULL);

    // load constant c into register
    const __m128 scalar = _mm_set1_ps(c);

    // start SIMD loop
    for (; xptr < aligned_bound; xptr += 4, yptr += 4, out += 4) {
        const __m128 xvec = _mm_loadu_ps(xptr);
        const __m128 yvec = _mm_loadu_ps(yptr);
        // _mm_storeu_ps(out, _mm_mul_ps(_mm_add_ps(xvec, scalar), yvec));
        _mm_storeu_ps(out, _mm_add_ps(_mm_mul_ps(xvec, scalar), yvec));
    }

    // handle remaining elements
    switch (n & 3ULL) {
        case 3: out[2] = xptr[2] * c + yptr[2]; [[fallthrough]];
        case 2: out[1] = xptr[1] * c + yptr[1]; [[fallthrough]];
        case 1: out[0] = xptr[0] * c + yptr[0];
        default: break;
    }
};

/**
 * @brief Performs SIMD-accelerated element-wise subtraction of two single-precision
 *        floating-point arrays.
 *
 * Computes out[i] = x[i] - y[i] for each element using SSE instructions.
 * Optimized for 32-bit floating-point data with automatic scalar fallback.
 *
 * @param x       [in] Pointer to first input array (minuend array)
 * @param y       [in] Pointer to second input array (subtrahend array)
 * @param n       [in] Number of elements to process (primary size parameter)
 * @param m       [in] Secondary size parameter (currently unused, reserved for future)
 * @param out     [out] Pointer to output array (must have capacity for at least n elements)
 *
 * @note Implementation Details:
 * - Uses SSE instructions to process 4 floats per operation when n ≥ 4
 * - Automatically falls back to scalar operations for small arrays (n < 4)
 * - Handles remaining elements (n % 4) after vector processing
 * - No-throw guarantee (noexcept qualified)
 * - Supports unaligned memory accesses
 * - Current implementation ignores parameter m
 *
 * @warning Undefined behavior if:
 * - Input pointers are null when n > 0
 * - Array sizes are smaller than n
 * - Input and output ranges overlap
 * - n doesn't match actual array dimensions
 *
 * @example Basic Usage:
 * std::vector<float> a(1000, 5.0f);
 * std::vector<float> b(1000, 3.0f);
 * std::vector<float> result(1000);
 * vecdiff_sse_float(a.data(), b.data(), a.size(), 0, result.data());
 * // result will contain 2.0f in each element
 */
inline void vecdiff_sse_float(const float* x,
                              const float* y,
                              const std::size_t n,
                              const std::size_t m,
                              float* out) noexcept {
    if (x == nullptr || y == nullptr) return ;
    if (out == nullptr) return ;
    if (n == 0 || m == 0) return ;
    if (m != n) return ;

    // handle small size n < 4
    if (n < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i] - y[i];
        }
        return ;
    }

    // define xptr and yptr point to x and y
    const float* xptr = x;
    const float* yptr = y;
    const float* aligned_bound = x + (n & ~3ULL);

    // start SIMD loop
    for (; xptr < aligned_bound; xptr += 4, yptr += 4, out += 4) {
        const __m128 xvec = _mm_loadu_ps(xptr);
        const __m128 yvec = _mm_loadu_ps(yptr);
        _mm_storeu_ps(out, _mm_sub_ps(xvec, yvec));
    }

    // handle remaining elements
    switch (n & 3ULL) {
        case 3: out[2] = xptr[2] - yptr[2]; [[fallthrough]];
        case 2: out[1] = xptr[1] - yptr[1]; [[fallthrough]];
        case 1: out[0] = xptr[0] - yptr[0];
        default: break;
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
 * - Uses SSE4.2 instruction set (requires CPU support)
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
inline float vecdot_sse_float(const float* x,
                              const float* y,
                              std::size_t n,
                              std::size_t m) noexcept {
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




}
}
#endif // MATH_MATH_KERNELS_OPS_SSE_OPS_SSE_FLOAT_HPP_
