#ifndef MATH_KERNELS_OPS_SSE_OPS_SSE_DOUBLE_HPP_
#define MATH_KERNELS_OPS_SSE_OPS_SSE_DOUBLE_HPP_

#include "common/consts.hpp"
#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

/**
 * @brief Sets all elements of a double array to a specified value using SSE instructions.
 *
 * @param[in,out] x Pointer to the double array to be filled.
 * @param[in] c The constant value to set all array elements to.
 * @param[in] n Number of elements in the array.
 *
 * @details This function efficiently fills a double array with a constant value using SSE SIMD instructions.
 * It handles:
 * - Null pointers and zero-length arrays (no-op)
 * - Small arrays (<4 elements) with scalar operations
 * - Handles both aligned and unaligned memory automatically
 * - Aligned blocks (4 elements per SSE operation)
 * - Remaining elements (tail handling)
 *
 * @note For best performance:
 * - Memory should be 16-byte aligned
 * - Array size should be reasonably large (>16 elements) to amortize setup costs
 * - Uses SSE intrinsics (_mm_set1_ps, _mm_store_ps)
 *
 * @exception None (no-throw guarantee)
 *
 * @example
 * // Example usage:
 * double data[16];
 * vecset_sse_double(data, 3.14f, 16); // Sets all 16 elements to 3.14f
 */
inline void vecset_sse_double(double* x, const double c, std::size_t n) noexcept {
    if (x == nullptr || n == 0) return ;

    // handle small size n < 2
    if (n < 2) {
        x[0] = c;
        return ;
    }

    // define a ptr points to x, end of bound
    double* xptr = x;
    const double* end = x + n;

    // handle unaligned case
    while (reinterpret_cast<std::uintptr_t>(xptr) & SSE_MEMOPS_ALIGNMENT) {
        *xptr = c;
        xptr++;
        if (xptr == end) return;
    }

    // handle aligned case
    // load scalar c into register and define aligned bound
    const __m128d scalar = _mm_set1_pd(c);
    const double* algned_end = xptr + ((end - xptr) & ~1ULL);
    for (; xptr < algned_end; xptr += 2) {
        _mm_store_pd(xptr, scalar);
    }

    // handle remaining elements
    if (end > xptr) {
        xptr[0] = c;
    }
}


/**
 * @brief Clips (clamps) an array of double-precision values to a specified range using SSE intrinsics.
 *
 * This function efficiently processes an array of doubles, ensuring all values fall within the [min, max] range.
 * It handles special cases including NaN values (preserving them), unaligned memory, and various edge conditions.
 *
 * @param[in,out] x Pointer to the array of double-precision values to be clipped.
 *                  Must be 16-byte aligned for optimal performance.
 * @param[in] min The minimum bound of the clipping range (inclusive).
 * @param[in] max The maximum bound of the clipping range (inclusive).
 * @param[in] n Number of elements in the array.
 *
 * @note Key features:
 * - Preserves NaN values in the input array
 * - Handles both aligned and unaligned memory automatically
 * - Uses SSE2 intrinsics for vectorized processing
 * - Processes 2 elements per iteration when aligned
 * - No-throw guarantee
 *
 * @performance For best performance:
 * - Memory should be 16-byte aligned
 * - Array size should be reasonably large (>8 elements) to amortize setup costs
 * - Uses _mm_set1_pd, _mm_load_pd, _mm_store_pd intrinsics
 *
 * @exception None (no-throw guarantee)
 *
 * @example
 * // Clip array values to [0.0, 1.0] range
 * double data[4] = {-1.0, 0.5, 2.0, NAN};
 * vecclip_sse_double(data, 0.0, 1.0, 4);
 * // data now contains [0.0, 0.5, 1.0, NAN]
 */
inline void vecclip_sse_double(double* x, double min, double max, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return ;
    if (min > max) return ;

    // Create an SSE register with all elements set to the min/max value
    const __m128d xmin = _mm_set1_pd(min);
    const __m128d xmax = _mm_set1_pd(max);

    // check if x is aligned to 16 bytes == 2 elems
    if (n < 2) {
        x[0] = std::clamp(x[0], min, max);
        return ;
    }

    // define ptr points to x and end of x
    double* xptr = x;
    const double* end = x + n;

    // handle unaligned case, 16-byte alignment
    while (reinterpret_cast<std::uintptr_t>(xptr) & SSE_MEMOPS_ALIGNMENT) {
        *xptr = std::clamp(*xptr, min, max);
        xptr++;
        if (xptr == end) return ;
    }

    // compute aligned bound
    const double* aligned_end = xptr + ((end - xptr) & ~1ULL);

    // Process the array in chunks of 2 elements using SSE intrinsics
    for (; xptr < aligned_end; xptr += 2) {
        __m128d vec = _mm_load_pd(xptr);

        // clamp the vector to the range [min, max]
        __m128d clipped = _mm_min_pd(_mm_max_pd(vec, xmin), xmax);

        // nan_mask: check nan value of vec, nan = 1, otherwise is 0
        // _mm_and_pd: keep nan val of vec, nan = 1, otherwise is 0
        // _mm_andnot_pd: keep non-nan val of clipped vec, nan = 0, otherwise is 1
        // _mm_or_pd: combine the two results
        __m128d nan_mask = _mm_cmpunord_pd(vec, vec);
        vec = _mm_or_pd(_mm_and_pd(nan_mask, vec),
                        _mm_andnot_pd(nan_mask, clipped));
        _mm_store_pd(xptr, vec);
    }

    // Process any remaining elements
    if (end > xptr) {
        xptr[0] = std::clamp(xptr[0], min, max);
    }
};

/**
 * @brief Checks if a double-precision array contains any infinite values using SSE intrinsics.
 *
 * @param[in] x Pointer to the array of double-precision values to check.
 *              Memory does not need to be aligned, but 16-byte alignment improves performance.
 * @param[in] n Number of elements in the array.
 *
 * @return true if any element is ±infinity, false otherwise or for empty/null input.
 *
 * @details This function efficiently scans an array for infinite values using SSE2 instructions.
 * Key features:
 * - Handles both positive and negative infinity
 * - Preserves NaN values (does not treat them as infinite)
 * - Automatic handling of unaligned memory
 * - Processes elements in chunks of 2 doubles (SSE register size)
 * - Special optimized paths for small arrays (n < 2)
 * - No-throw guarantee
 *
 * @performance For optimal performance:
 * - Memory should be 16-byte aligned
 * - Large arrays (n > 8) benefit most from vectorization
 * - Uses _mm_set1_pd, _mm_cmpeq_pd, _mm_or_pd intrinsics
 *
 * @exception None (no-throw guarantee)
 *
 * @example
 * // Check array with finite values
 * double data1[4] = {1.0, 2.0, 3.0, 4.0};
 * bool has_inf1 = hasinf_sse_double(data1, 4); // returns false
 *
 * // Check array with infinity
 * double data2[4] = {1.0, INFINITY, 3.0, NAN};
 * bool has_inf2 = hasinf_sse_double(data2, 4); // returns true
 */
inline bool hasinf_sse_double(const double* x, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return false;

    // check if x has 2 elems
    if (n < 2) {
        return std::isinf(x[0]) ? true : false;
    }

    // define ptr points to x and end of x
    const double* xptr = x;
    const double* end = x + n;

    // handle unaligned case, 16-byte alignment
    while (reinterpret_cast<std::uintptr_t>(xptr) & SSE_MEMOPS_ALIGNMENT) {
        if (std::isinf(*xptr)) return true;
        xptr++;
        if (xptr == end) return false;
    }

    // load x positif inf + x negative inf to SIMD register
    const __m128d pos_inf = _mm_set1_pd(std::numeric_limits<double>::infinity());
    const __m128d neg_inf = _mm_set1_pd(-std::numeric_limits<double>::infinity());

    // compute aligned bound
    const double* aligned_end = xptr + ((end - xptr) & ~1ULL);

    // loop the array in chunks of 2 elements
    for (; xptr < aligned_end; xptr += 2) {
        const __m128d vec = _mm_load_pd(xptr);
        const __m128d cmp = _mm_or_pd(_mm_cmpeq_pd(vec, pos_inf),
                                      _mm_cmpeq_pd(vec, neg_inf));
        if (_mm_movemask_pd(cmp) != 0) {
            return true;
        }
    }

    // process remaining elements
    return (end > xptr) ? std::isinf(xptr[0]) : false;
};

/**
 * @brief Computes the squared L2 norm (Euclidean norm) of a double-precision vector using SSE intrinsics.
 *
 * @param[in] x Pointer to the array of double-precision values. Must be 16-byte aligned for optimal performance.
 * @param[in] n Number of elements in the array.
 * @param[in] squared If true, returns the squared norm (faster); if false, returns the actual L2 norm.
 *
 * @return The computed L2 norm (or squared L2 norm if requested).
 *
 * @details This function efficiently calculates the Euclidean norm of a vector using SSE SIMD instructions.
 * Features:
 * - Handles both aligned and unaligned memory automatically
 * - Processes elements in chunks of 2 doubles (SSE register size)
 * - Optional squared output avoids final square root for faster computation
 * - Special handling for small vectors (n < 2)
 * - No-throw guarantee
 *
 * @note For best performance:
 * - Memory should be 16-byte aligned
 * - Large vectors (n > 8) will benefit most from vectorization
 * - Uses _mm_load_pd, _mm_mul_pd, _mm_hadd_pd intrinsics
 *
 * @exception None (no-throw guarantee)
 *
 * @example
 * // Compute regular L2 norm
 * double data[4] = {1.0, 2.0, 3.0, 4.0};
 * double norm = vecnorm2_sse_double(data, 4, false);
 * // norm = √(1² + 2² + 3² + 4²) = 5.477...
 *
 * // Compute squared norm (faster)
 * double norm_sq = vecnorm2_sse_double(data, 4, true);
 * // norm_sq = 1² + 2² + 3² + 4² = 30.0
 */
inline double vecnorm2_sse_double(const double* x,
                                  std::size_t n,
                                  bool squared) noexcept {
    if (n == 0 || x == nullptr) return 0.0;
    // For small arrays with less than or equal to two elements
    if (n < 2) {
        double sum = 0.0;
        sum += x[0] * x[0];
        return squared ? sum : std::sqrt(sum);
    }

    // compute aligned boundary
    const double* xptr = x;
    const double* end = x + n;

    // handle unaligned memeroy case
    double total = 0.0;
    while (reinterpret_cast<std::uintptr_t>(xptr) & SSE_MEMOPS_ALIGNMENT) {
        total += (*xptr) * (*xptr);
        xptr++;
        if (xptr == end) return squared ? total : std::sqrt(total);
    }

    // handle aligned case
    const double* aligned_end = xptr + ((end - xptr) & ~1ULL);

    // init sum to 0
    __m128d sum = _mm_setzero_pd();
    for (; xptr < aligned_end; xptr += 2) {
        // load 2 elements from x to SIMD register
        const __m128d vec = _mm_load_pd(xptr);
        // compute sum = sum + vec * vec
        sum = _mm_add_pd(sum, _mm_mul_pd(vec, vec));
    }

    // perform a horizontal addition of the two channels' values in the SSE register
    __m128d sumh = _mm_hadd_pd(sum, sum);
    total += _mm_cvtsd_f64(sumh);

    // process remaining elements
    if (end > xptr) {
        total += xptr[0] * xptr[0];
    }
    return squared ? total : std::sqrt(total);
};

/**
 * @brief Computes the L1 norm (sum of absolute values) of a double-precision
 *        array using SSE intrinsics.
 *
 * @param[in] x  Pointer to input array (must be non-null unless n=0)
 * @param[in] n  Number of elements in array
 * @return       The computed L1 norm (Σ|x[i]|)
 *
 * @implementation
 *   - Vectorized SSE implementation (2 elements per SIMD operation)
 *   - Special handling for small arrays (n < 2)
 *   - Correctly handles remaining elements after SIMD loop
 *
 * @requirements
 *   - SSE4.2 instruction set (-msse4.2)
 *   - 16-byte alignment recommended for best performance
 *
 * @exception noexcept guaranteed
 * @complexity O(n) with ~2x speedup vs scalar code
 *
 * Example:
 *   double data[4] = {1.0, -2.0, 3.0, -4.0};
 *   double norm = vecnorm1_sse_double(data, 4); // Returns 10.0
 */
inline double vecnorm1_sse_double(const double* x,
                                  std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return 0.0;
    // For small arrays with less than or equal to two elements
    if (n < 2) {
        double sum = 0.0;
        sum += std::abs(x[0]);
        return sum;
    }

    // define ptr points to x and end of x
    const double* xptr = x;
    const double* end = x + n;

    // handle case of memory unaligned
    double total = 0.0;
    while (reinterpret_cast<std::uintptr_t>(xptr) & SSE_MEMOPS_ALIGNMENT) {
        total += std::abs(*xptr);
        xptr++;
        if (xptr == end) return total;
    }

    // compute aligned bound
    const double* aligned_end = xptr + ((end - xptr) & ~1ULL);

    // mask for the absolute value of the a double
    const __m128d abs_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));

    // init sum to 0
    __m128d sum = _mm_setzero_pd();
    for (; xptr < aligned_end; xptr += 2) {
        // load 2 elements from x to SIMD register
        __m128d vec = _mm_load_pd(xptr);
        sum = _mm_add_pd(sum, _mm_and_pd(vec, abs_mask));
    }

    // perform a horizontal addition of the two channels' values in the SSE register
    __m128d sumh = _mm_hadd_pd(sum, sum);
    // _mm_store_sd(&total, sumh);
    total += _mm_cvtsd_f64(sumh);

    // handle remaining elements
    if (end > xptr) {
        total += std::abs(xptr[0]);
    }

    return total;
};

/**
 * @brief Performs in-place scaling of a double-precision array using SSE4.2 intrinsics.
 *
 * Computes: x[i] = x[i] * c  for all i in [0, n-1]
 *
 * @param[in,out] x  Pointer to the array to be scaled (modified in-place)
 * @param[in]     n  Number of elements in the array
 * @param[in]     c  Scaling factor
 * @param[out]    out Output vector (same size with x)
 *
 * @details Features:
 *   - Vectorized using SSE4.2 (processes 4.2 doubles per SIMD operation)
 *   - Handles unaligned memory accesses safely
 *   - Early exit if:
 *       - x == nullptr
 *       - c == 1.0 (identity operation)
 *   - Processes remaining elements (n % 2 != 0) after SIMD loop
 *
 * @requirements:
 *   - SSE4.2 instruction set (-msse4.2 compiler flag)
 *   - 16-byte alignment recommended for optimal performance
 *
 * @exception noexcept guaranteed
 * @complexity O(n) with ~2x speedup vs scalar implementation
 *
 * @example:
 *   double data[1000];
 *   vecscale_sse_double(data, 3.14, 1000, out); // data *= 3.14
 */
inline void vecscale_sse_double(const double* x,
                                const double c,
                                std::size_t n,
                                double* out) noexcept {
    if (x == nullptr || out == nullptr) return ;
    if (n == 0 || c == 1.0f) return ;

    // for small size n < 2
    if (n < 2) {
        out[0] = x[0] * c;
        return ;
    }

     // define ptr points to x and end of x
     double* outptr = out;
     const double* xptr = x;
     const double* end = x + n;

     // handle case of memory unaligned
    while (reinterpret_cast<std::uintptr_t>(xptr) & SSE_MEMOPS_ALIGNMENT ||
           reinterpret_cast<std::uintptr_t>(outptr) & SSE_MEMOPS_ALIGNMENT) {
        *outptr = *xptr * c;
        xptr++;
        outptr++;
        if (xptr == end) return ;
    }

    // define ptr points to x and aligned end
    const double* aligned_end = xptr + ((end - xptr) & ~1ULL);

    // load constant into register
    const __m128d scalar = _mm_set1_pd(c);

    // main SIMD loop
    for (; xptr < aligned_end; xptr += 2, outptr += 2) {
        const __m128d xvec = _mm_load_pd(xptr);
        // const __m128d outx = _mm_mul_pd(xvec, scalar);
        _mm_store_pd(outptr, _mm_mul_pd(xvec, scalar));
    }

    // handle remaining elements
    if (end > xptr) {
        outptr[0] = xptr[0] * c;
    }
};

/**
 * @brief Performs SIMD-accelerated vector scaling (multiplication by constant) for double-precision arrays.
 *
 * Computes out[i] = x[i] * c for each element in the range [xbegin, xend) using SSE instructions.
 * Optimized for double-precision floating-point data with automatic fallback to scalar operations.
 *
 * @param[in] xbegin  Pointer to the first element of the input array (must be valid if n > 0)
 * @param[in] xend    Pointer to one past the last element of the input array
 * @param[in] c       Scaling constant to multiply each element by
 * @param[in] n       Number of elements to process (must equal distance between xbegin and xend)
 * @param[out] out    Pointer to the output array (must have capacity for at least n elements)
 *
 * @note Features:
 * - Uses SSE4.2 instructions for processing 2 doubles at a time when n >= 2
 * - Falls back to scalar operations for small arrays (n < 2)
 * - Handles remaining elements (n % 2) after SIMD processing
 * - No-throw guarantee (noexcept)
 * - Memory-safe: checks pointer validity before access
 *
 * @warning Behavior is undefined if:
 * - xbegin and xend don't form a valid range (xend >= xbegin)
 * - n doesn't match the actual array size
 * - Input and output ranges overlap
 * - Pointers are null when n > 0
 *
 * @example
 * std::vector<double> x(100, 2.0);
 * std::vector<double> out(100);
 * vecscale_sse_double(x.data(), x.data() + x.size(), 3.14, x.size(), out.data());
 */
inline void vecscale_sse_double(const double* xbegin,
                                const double* xend,
                                const double c,
                                std::size_t n,
                                double* out) noexcept {
    if (xbegin == nullptr || xend == nullptr || out == nullptr) return;
    if (n == 0 || c == 1.0) return ;
    if (xend <= xbegin) return ;
    const std::size_t m = static_cast<std::size_t>(xend - xbegin);
    if (n != m) return ;

    // call vecscale_sse_double function
    vecscale_sse_double(xbegin, c, n, out);
};

/**
 * @brief Performs SIMD-accelerated element-wise addition of two
 *        double-precision arrays.
 *
 * Computes out[i] = x[i] + y[i] for each element using SSE2 instructions.
 * Optimized for 64-bit floating-point data with automatic scalar fallback.
 *
 * @param x       [in] Pointer to first input array (must not be null if n > 0)
 * @param y       [in] Pointer to second input array (must not be null if n > 0)
 * @param n       [in] Number of elements to process (primary size parameter)
 * @param m       [in] Secondary size parameter (currently unused, reserved for
 *                     future compatibility)
 * @param out     [out] Pointer to output array (must have capacity for n elements)
 *
 * @note Features:
 * - Uses SSE4.2 instructions to process 2 doubles per operation when n ≥ 2
 * - Falls back to scalar processing for small arrays (n < 2)
 * - Handles remaining elements (n % 2) after vector processing
 * - No-throw guarantee (noexcept qualified)
 * - Supports unaligned memory accesses
 * - Parameter m is currently unused but maintained for interface consistency
 *
 * @warning Undefined behavior if:
 * - Input pointers are null when n > 0
 * - Array sizes are smaller than n
 * - Input and output ranges overlap
 * - n doesn't match actual array dimensions
 *
 * @example Typical Usage:
 * std::vector<double> a(1000, 1.0);
 * std::vector<double> b(1000, 2.0);
 * std::vector<double> result(1000);
 * vecadd_sse_double(a.data(), b.data(), a.size(), 0, result.data());
 */
inline void vecadd_sse_double(const double* x,
                              const double* y,
                              const std::size_t n,
                              const std::size_t m,
                              double* out) noexcept {
    if (x == nullptr || y == nullptr) return ;
    if (out == nullptr) return ;
    if (n == 0 || m == 0) return ;
    if (m != n) return ;

    if (n < 2) {
        out[0] = x[0] + y[0];
        return ;
    }

    // define ptr points to x and aligned end
    double* outptr = out;
    const double* xptr = x;
    const double* yptr = y;
    const double* end = x + n;

    // handle memory unaligned case
    while (reinterpret_cast<std::uintptr_t>(xptr) & SSE_MEMOPS_ALIGNMENT ||
           reinterpret_cast<std::uintptr_t>(yptr) & SSE_MEMOPS_ALIGNMENT ||
           reinterpret_cast<std::uintptr_t>(outptr) & SSE_MEMOPS_ALIGNMENT) {
        *outptr = *xptr + *yptr;
        xptr++;
        yptr++;
        outptr++;
        if (xptr == end || yptr == end) return ;
    }

    // define aligned bound
    const double* aligned_end = xptr + ((end - xptr) & ~1ULL);

    for (; xptr < aligned_end; xptr += 2, yptr += 2, outptr += 2) {
        const __m128d xvec = _mm_load_pd(xptr);
        const __m128d yvec = _mm_load_pd(yptr);
        _mm_store_pd(outptr, _mm_add_pd(xvec, yvec));
    }

    if (end > xptr) {
        outptr[0] = xptr[0] + yptr[0];
    }
};

/**
 * @brief Performs SIMD-accelerated vector addition with scaling for double-precision arrays.
 *
 * Computes the element-wise operation: out[i] = x[i] * c + y[i] using SSE2 instructions.
 * Optimized for 64-bit floating-point data with automatic scalar fallback.
 *
 * @param x       [in] Pointer to first input array (must be valid if n > 0)
 * @param y       [in] Pointer to second input array (must be valid if n > 0)
 * @param c       [in] Scaling factor to apply to x elements before addition
 * @param n       [in] Number of elements to process (primary size parameter)
 * @param m       [in] Secondary size parameter (currently unused, reserved for future extension)
 * @param out     [out] Pointer to output array (must have capacity for n elements)
 *
 * @note Implementation Details:
 * - Uses SSE4.2 instructions to process 2 doubles per operation (when n ≥ 2)
 * - Broadcasts scaling factor c to SSE register for vectorized multiplication
 * - Falls back to scalar processing for small arrays (n < 2)
 * - Handles remaining elements (n % 2) after vector processing
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
 * std::vector<double> a(1000, 1.0);
 * std::vector<double> b(1000, 2.0);
 * std::vector<double> result(1000);
 * vecadd_sse_double(a.data(), b.data(), 3.14, a.size(), 0, result.data());
 */
inline void vecadd_sse_double(const double* x,
                              const double* y,
                              const double c,
                              const std::size_t n,
                              const std::size_t m,
                              double* out) noexcept {
    if (x == nullptr || y == nullptr) return ;
    if (out == nullptr) return ;
    if (n == 0 || m == 0) return ;
    if (m != n) return ;

    if (n < 2) {
        out[0] = x[0] * c + y[0];
        return ;
    }

    // load constant c into register
    const __m128d scalar = _mm_set1_pd(c);

    // define ptr points to x and aligned end
    const double* xptr = x;
    const double* yptr = y;
    const double* aligned_bound = x + (n & ~1ULL);

    for (; xptr < aligned_bound; xptr += 2, yptr += 2, out += 2) {
        const __m128d xvec = _mm_loadu_pd(xptr);
        const __m128d yvec = _mm_loadu_pd(yptr);
        _mm_storeu_pd(out, _mm_add_pd(_mm_mul_pd(xvec, scalar), yvec));
    }

    if (n & 1ULL) {
        out[0] = xptr[0] * c + yptr[0];
    }
}

/**
 * @brief Performs SIMD-accelerated element-wise subtraction of two double-precision
 *        floating-point arrays.
 *
 * Computes out[i] = x[i] - y[i] for each element using SSE4.2 instructions.
 * Optimized for 64-bit floating-point data with automatic scalar fallback.
 *
 * @param[in] x      Pointer to first input array (minuend array)
 * @param[in] y      Pointer to second input array (subtrahend array)
 * @param[in] n      Number of elements to process (primary size parameter)
 * @param[in] m      Secondary size parameter (currently unused, reserved for future)
 * @param[out] out   Pointer to output array (must have capacity for at least n elements)
 *
 * @note Implementation Details:
 * - Uses SSE4.2 instructions to process 2 double per operation when n ≥ 2
 * - Automatically falls back to scalar operations for small arrays (n < 2)
 * - Handles remaining elements (n % 2) after vector processing
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
 * std::vector<double> a(1000, 5.0);
 * std::vector<double> b(1000, 3.0);
 * std::vector<double> result(1000);
 * vecdiff_sse_double(a.data(), b.data(), a.size(), 0, result.data());
 * // result will contain 2.0f in each element
 */
inline void vecdiff_sse_double(const double* x,
                               const double* y,
                               const std::size_t n,
                               const std::size_t m,
                               double* out) noexcept {
    if (x == nullptr || y == nullptr) return ;
    if (out == nullptr) return ;
    if (n == 0 || m == 0) return ;
    if (m != n) return ;

    if (n < 2) {
        out[0] = x[0] - y[0];
        return ;
    }

    // define ptr points to x and aligned end
    const double* xptr = x;
    const double* yptr = y;
    const double* aligned_bound = x + (n & ~1ULL);

    for (; xptr < aligned_bound; xptr += 2, yptr += 2, out += 2) {
        const __m128d xvec = _mm_loadu_pd(xptr);
        const __m128d yvec = _mm_loadu_pd(yptr);
        _mm_storeu_pd(out, _mm_sub_pd(xvec, yvec));
    }

    if (n & 1ULL) {
        out[0] = xptr[0] - yptr[0];
    }
};

/**
 * @brief Computes the dot product of two double-precision vectors using SSE4.2 intrinsics.
 *
 * Calculates the sum of element-wise products: Σ(x[i] * y[i]) for i = 0 to n-1.
 * Optimized for 64-bit floating-point data with automatic scalar fallback.
 *
 * @param[in] x   Pointer to first input vector (must be valid if n > 0)
 * @param[in] y   Pointer to second input vector (must be valid if n > 0)
 * @param[in] n   Number of elements to process (primary size parameter)
 * @param[in] m   Secondary size parameter (currently unused, reserved for future)
 * @return        The computed dot product as a double-precision value
 *
 * @note Implementation Details:
 * - Uses SSE2 instructions to process 2 doubles per operation when n ≥ 2
 * - Maintains partial sums in SSE registers for better precision
 * - Automatically falls back to scalar operations for small vectors (n < 2)
 * - No-throw guarantee (noexcept qualified)
 * - Supports unaligned memory accesses
 * - Current implementation ignores parameter m
 *
 * @warning Undefined behavior if:
 * - Input pointers are null when n > 0
 * - Vector sizes are smaller than n
 * - Parameter n doesn't match actual array dimensions
 *
 * @example Basic Usage:
 * std::vector<double> a = {1.0, 2.0, 3.0};
 * std::vector<double> b = {4.0, 5.0, 6.0};
 * double result = vecdot_sse_double(a.data(), b.data(), a.size(), 0);
 * // result = 1.0*4.0 + 2.0*5.0 + 3.0*6.0 = 32.0
 */
inline double vecdot_sse_double(const double* x,
                                const double* y,
                                std::size_t n,
                                std::size_t m) noexcept {
    if (x == nullptr || y == nullptr) return 0.0;
    if (n == 0 || m == 0) return 0.0;
    if (n != m) return 0.0;

    // for small size array
    if (n < 2) {
        double sum = 0.0;
        sum += x[0] * y[0];
        return sum;
    }

    const double* xptr = x;
    const double* yptr = y;
    const double* aligned_bound = x + (n & ~1ULL);

    // load sum of vec to register
    __m128d sum = _mm_setzero_pd();

    for (; xptr < aligned_bound; xptr += 2, yptr += 2) {
        const __m128d xvec = _mm_loadu_pd(xptr);
        const __m128d yvec = _mm_loadu_pd(yptr);
        sum = _mm_add_pd(sum, _mm_mul_pd(xvec, yvec));
    }

    double total;
    // Extract high element of sum register
    // add low and high
    __m128d sumh = _mm_unpackhi_pd(sum, sum);
    __m128d tmp = _mm_add_pd(sum, sumh);
    _mm_store_sd(&total, tmp);

    if (n & 1ULL) {
        total += x[n - 1] * y[n - 1];
    }
    return total;
};





}
}
#endif // MATH_KERNELS_OPS_SSE_OPS_SSE_DOUBLE_HPP_
