#ifndef MATH_MATH_KERNELS_OPS_SSE_OPS_SSE_DOUBLE_HPP_
#define MATH_MATH_KERNELS_OPS_SSE_OPS_SSE_DOUBLE_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

inline void vecset_sse_double(double* x, const double c, std::size_t n) noexcept {
    if (x == nullptr || n == 0) return ;
    if (n < 2) {
        x[0] = c;
        return ;
    }

    // compute aligned bound
    const double* ptr = x;
    const double* algned_bound = x + (n & ~1ULL);

    const __mm128d scalar = _mm_set1_pd(c);

    for (; ptr < algned_bound; ptr += 2) {
        _mm_storeu_pd(ptr, scalar);
    }

    if (n & 1ULL) {
        ptr[0] = c;
    }
}




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
};

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
inline bool hasinf_sse_double(const double* x, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return false;

    // check if x has 2 elems
    if (n < 2) {
        return std::isinf(x[0]) ? true : false;
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
};

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
 * @param squared If `true`, returns the squared L2 norm; if `false`,
 *                returns the square root of the L2 norm.
 * @return double The computed L2 norm (either squared or its square root
 *         depending on the `squared` parameter).
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

    // compute aligned bound = xsize - xsize % 2
    const double* ptr = x;
    const double* aligned_bound = x + (n & ~1ULL);

    // mask for the absolute value of the a double
    const __m128d abs_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));

    // init sum to 0
    __m128d sum = _mm_setzero_pd();
    for (; ptr < aligned_bound; ptr += 2) {
        // load 2 elements from x to SIMD register
        __m128d vec = _mm_loadu_pd(ptr);
        sum = _mm_add_pd(sum, _mm_and_pd(vec, abs_mask));
    }

    double total;
    // perform a horizontal addition of the two channels' values in the SSE register
    __m128d sumh = _mm_hadd_pd(sum, sum);
    _mm_store_sd(&total, sumh);

    // handle remaining elements
    if (n & 1ULL) {
        total += std::abs(x[n - 1]);
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

    // for small size n <= 2
    if (n < 2) {
        out[0] = x[0] * c;
        return ;
    }

    // define ptr points to x and aligned end
    const double* ptr = x;
    const double* aligned_bound = x + (n & ~1ULL);

    // load constant into register
    const __m128d scalar = _mm_set1_pd(c);

    // main SIMD loop
    for (; ptr < aligned_bound; ptr += 2, out += 2) {
        const __m128d xvec = _mm_loadu_pd(ptr);
        const __m128d outx = _mm_mul_pd(xvec, scalar);
        _mm_storeu_pd(out, outx);
    }

    // handle remaining elements
    if (n & 1ULL) {
        out[0] = ptr[0] * c;
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
    const double* xptr = x;
    const double* yptr = y;
    const double* aligned_bound = x + (n & ~1ULL);

    for (; xptr < aligned_bound; xptr += 2, yptr += 2, out += 2) {
        const __m128d xvec = _mm_loadu_pd(xptr);
        const __m128d yvec = _mm_loadu_pd(yptr);
        _mm_storeu_pd(out, _mm_add_pd(xvec, yvec));
    }

    if (n & 1ULL) {
        out[0] = xptr[0] + yptr[0];
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
#endif // MATH_MATH_KERNELS_OPS_SSE_OPS_SSE_DOUBLE_HPP_
