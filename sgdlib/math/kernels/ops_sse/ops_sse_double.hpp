#ifndef MATH_KERNELS_OPS_SSE_OPS_SSE_DOUBLE_HPP_
#define MATH_KERNELS_OPS_SSE_OPS_SSE_DOUBLE_HPP_

#include "common/constants.hpp"
#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

#if defined(USE_SSE)

/**
 * @brief Sets all elements of a double array to a specified value using SSE instructions.
 */
inline void vecset_sse_double(double* x, const double c, std::size_t n) noexcept {
    // define a ptr points to x, end of bound
    double* xptr = x;
    const double* end = x + n;
    const __m128d scalar = _mm_set1_pd(c);

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // main loop to handle 4 elements
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        _mm_storeu_pd(xptr, scalar);
        _mm_storeu_pd(xptr + 2, scalar);
        xptr += DTYPE_UNROLLING_SIZE;
    }

    // handle remaining elements by 4
    const std::size_t remainder =  end - xptr;
    switch (remainder) {
        case 3: xptr[2] = c; [[fallthrough]];
        case 2: xptr[1] = c; [[fallthrough]];
        case 1: xptr[0] = c;
        default: break;
    }
};

/**
 * @brief Copies an array of double-precision floating-point numbers using SSE4.2 instructions.
 */
void veccpy_sse_double(const double* x, std::size_t n, double* out) noexcept {
    // define xptr, outptr and a ptr to end of x
    double* outptr = out;
    const double* xptr = x;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // define aligned bound
    const double* aligned_end = xptr + ((end - xptr) & ~1ULL);

    // main loop to process simd
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m128d xvec0 = _mm_loadu_pd(xptr);
        __m128d xvec1 = _mm_loadu_pd(xptr + 2);
        _mm_storeu_pd(outptr, xvec0);
        _mm_storeu_pd(outptr + 2, xvec1);
        xptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }

    // handle remaining elements
    const std::size_t remainder = end - xptr;
    switch (remainder) {
        case 3: outptr[2] = xptr[2]; [[fallthrough]];
        case 2: outptr[1] = xptr[1]; [[fallthrough]];
        case 1: outptr[0] = xptr[0];
        default: break;
    }
};

/**
 * @brief Negates and copies elements of a double array using SSE intrinsics.
 */
void vecncpy_sse_double(const double* x, std::size_t n, double* out) noexcept {
    // define xptr, outptr and a ptr to end of x
    double* outptr = out;
    const double* xptr = x;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // define sign_flip mask with 64bit
    const __m128d sign_flip = _mm_castsi128_pd(
        _mm_set1_epi64x(0x8000000000000000)
    );

    // main loop to process simd
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m128d xvec0 = _mm_loadu_pd(xptr);
        __m128d xvec1 = _mm_loadu_pd(xptr + 2);
        xvec0 = _mm_xor_pd(xvec0, sign_flip);
        xvec1 = _mm_xor_pd(xvec1, sign_flip);
        // put back to out pointer
        _mm_storeu_pd(outptr, xvec0);
        _mm_storeu_pd(outptr + 2, xvec1);
        // increments
        xptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }

    // handle remaining elements
    const std::size_t remainder = end - xptr;
    switch (remainder){
        case 3: outptr[2] = -xptr[2]; [[fallthrough]];
        case 2: outptr[1] = -xptr[1]; [[fallthrough]];
        case 1: outptr[0] = -xptr[0];
        default: break;
    }
};

/**
 * @brief Clips (clamps) an array of double-precision values to a specified range using SSE intrinsics.
 *        This function efficiently processes an array of doubles, ensuring all values fall
 *        within the [min, max] range. It handles special cases including NaN values (preserving them),
 *        unaligned memory, and various edge conditions.
 */
inline void vecclip_sse_double(double* x, double min, double max, std::size_t n) noexcept {
    // define ptr points to x and end of x
    double* xptr = x;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

     // Create an SSE register with all elements set to the min/max value
    const __m128d xmin = _mm_set1_pd(min);
    const __m128d xmax = _mm_set1_pd(max);

    // Process the array in chunks of 2 elements using SSE intrinsics
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m128d xvec0 = _mm_loadu_pd(xptr);
        __m128d xvec1 = _mm_loadu_pd(xptr + 2);

        // clamp the vector to the range [min, max]
        __m128d clipped1 = _mm_min_pd(_mm_max_pd(xvec0, xmin), xmax);
        __m128d clipped2 = _mm_min_pd(_mm_max_pd(xvec1, xmin), xmax);

        // nan_mask: check nan value of vec, nan = 1, otherwise is 0
        // _mm_and_pd: keep nan val of vec, nan = 1, otherwise is 0
        // _mm_andnot_pd: keep non-nan val of clipped vec, nan = 0, otherwise is 1
        // _mm_or_pd: combine the two results
        __m128d nan_mask1 = _mm_cmpunord_pd(xvec0, xvec0);
        __m128d nan_mask2 = _mm_cmpunord_pd(xvec1, xvec1);
        xvec0 = _mm_or_pd(_mm_and_pd(nan_mask1, xvec0),
                          _mm_andnot_pd(nan_mask1, clipped1));
        xvec1 = _mm_or_pd(_mm_and_pd(nan_mask2, xvec1),
                          _mm_andnot_pd(nan_mask2, clipped2));
        _mm_storeu_pd(xptr, xvec0);
        _mm_storeu_pd(xptr + 2, xvec1);

        xptr += DTYPE_UNROLLING_SIZE;
    }

    // Process any remaining elements
    // if (end > xptr) {
    //     xptr[0] = std::clamp(xptr[0], min, max);
    // }
    const std::size_t remainder = end - xptr;
    switch (remainder) {
        case 3: xptr[2] = std::clamp(xptr[2], min, max); [[fallthrough]];
        case 2: xptr[1] = std::clamp(xptr[1], min, max); [[fallthrough]];
        case 1: xptr[0] = std::clamp(xptr[0], min, max);
        default: break;
    }
};

/**
 * @brief Checks if a double-precision array contains any infinite values using SSE intrinsics.
 */
inline bool hasinf_sse_double(const double* x, std::size_t n) noexcept {
    // define ptr points to x and end of x
    const double* xptr = x;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // load x positif inf + x negative inf to SIMD register
    const __m128d pos_inf = _mm_set1_pd(INF);
    const __m128d neg_inf = _mm_set1_pd(-INF);

    // compute aligned bound
    // const double* aligned_end = xptr + ((end - xptr) & ~1ULL);

    // loop the array in chunks of 2 elements
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m128d xvec0 = _mm_loadu_pd(xptr);
        __m128d xvec1 = _mm_loadu_pd(xptr + 2);
        __m128d cmp0 = _mm_or_pd(_mm_cmpeq_pd(xvec0, pos_inf),
                                 _mm_cmpeq_pd(xvec0, neg_inf));
        __m128d cmp1 = _mm_or_pd(_mm_cmpeq_pd(xvec1, pos_inf),
                                 _mm_cmpeq_pd(xvec1, neg_inf));
        if (_mm_movemask_pd(_mm_or_pd(cmp0, cmp1)) != 0) {
            return true;
        }
    }

    // process remaining elements
    const std::size_t remainder = end - xptr;
    for (std::size_t i = 0; i < remainder; ++i) {
        if (std::isinf(xptr[i])) {
            return true;
        }
    }
    return false;
};

/**
 * @brief Computes the squared L2 norm (Euclidean norm) of
 *        a double-precision vector using SSE intrinsics.
 */
inline double vecnorm2_sse_double(const double* x,
                                  std::size_t n,
                                  bool squared) noexcept {
    const double* xptr = x;
    const double* end = x + n;
    double total = 0.0;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // init sum0 and sum1 to 0
    __m128d sum0 = _mm_setzero_pd();
    __m128d sum1 = _mm_setzero_pd();
    // main loop
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        // load 4 elements from x to 2 register
        __m128d xvec0 = _mm_loadu_pd(xptr);
        __m128d xvec1 = _mm_loadu_pd(xptr + 2);
        // compute sum = sum + vec * vec
        sum0 = _mm_add_pd(sum0, _mm_mul_pd(xvec0, xvec0));
        sum1 = _mm_add_pd(sum1, _mm_mul_pd(xvec1, xvec1));
        xptr += DTYPE_UNROLLING_SIZE;
    }

    // perform a horizontal addition
    // method 1:
    // const __m128d sumh1 = _mm_hadd_pd(sum0, sum0);
    // const __m128d sumh2 = _mm_hadd_pd(sum1, sum1);
    // const __m128d sumh = _mm_add_pd(sumh1, sumh2);
    const __m128d combine_sum = _mm_add_pd(sum0, sum1);
    const __m128d scalar_sum = _mm_add_pd(combine_sum, _mm_shuffle_pd(combine_sum, combine_sum, 0x01));
    total += _mm_cvtsd_f64(scalar_sum);

    // handle remaining elements
    const std::size_t remainder = end - xptr;
    switch (remainder) {
        case 3: total += xptr[2] * xptr[2]; [[fallthrough]];
        case 2: total += xptr[1] * xptr[1]; [[fallthrough]];
        case 1: total += xptr[0] * xptr[0];
        default: break;
    }
    return squared ? total : std::sqrt(total);
};

/**
 * @brief Computes the L1 norm (sum of absolute values) of a double-precision
 *        array using SSE4.2 intrinsics.
 */
inline double vecnorm1_sse_double(const double* x,
                                  std::size_t n) noexcept {
    // define ptr points to x and end of x
    const double* xptr = x;
    const double* end = x + n;
    double total = 0.0;

     // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // mask for the absolute value of the a double
    const __m128d abs_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));

    // init sum0 and sum1 to 0
    __m128d sum0 = _mm_setzero_pd();
    __m128d sum1 = _mm_setzero_pd();
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        // load 4 elements from x to 2 register
        __m128d xvec0 = _mm_loadu_pd(xptr);
        __m128d xvec1 = _mm_loadu_pd(xptr + 2);
        sum0 = _mm_add_pd(sum0, _mm_and_pd(xvec0, abs_mask));
        sum1 = _mm_add_pd(sum1, _mm_and_pd(xvec1, abs_mask));
        // increment
        xptr += DTYPE_UNROLLING_SIZE;
    }

    // perform a horizontal addition
    // the add 2 results of h-addition
    // const __m128d sumh1 = _mm_hadd_pd(sum0, sum0);
    // const __m128d sumh2 = _mm_hadd_pd(sum1, sum1);
    // const __m128d sumh = _mm_add_pd(sumh1, sumh2);
    const __m128d combine_sum = _mm_add_pd(sum0, sum1);
    const __m128d scalar_sum = _mm_add_pd(combine_sum, _mm_shuffle_pd(combine_sum, combine_sum, 0x01));
    total += _mm_cvtsd_f64(scalar_sum);

    // handle remaining elements
    const std::size_t remainder = end - xptr;
    switch (remainder) {
        case 3: total += std::abs(xptr[2]); [[fallthrough]];
        case 2: total += std::abs(xptr[1]); [[fallthrough]];
        case 1: total += std::abs(xptr[0]);
        default: break;
    }
    return total;
};

/**
 * @brief Performs in-place scaling of a double-precision array using SSE4.2 intrinsics.
 */
inline void vecscale_sse_double(const double* x,
                                const double c,
                                std::size_t n,
                                double* out) noexcept {
    // define ptr points to x and end of x
    double* outptr = out;
    const double* xptr = x;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // load constant into register
    const __m128d scalar = _mm_set1_pd(c);
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        // load xptr and xptr + 2
        __m128d xvec0 = _mm_loadu_pd(xptr);
        __m128d xvec1 = _mm_loadu_pd(xptr + 2);
        // put results x * c back to out pointer
        _mm_storeu_pd(outptr, _mm_mul_pd(xvec0, scalar));
        _mm_storeu_pd(outptr + 2, _mm_mul_pd(xvec1, scalar));
        // increment by DTYPE_UNROLLING_SIZE
        xptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }

    // handle remaining elements
    const std::size_t remainder = end - xptr;
    switch (remainder) {
        case 3: outptr[2] = xptr[2] * c; [[fallthrough]];
        case 2: outptr[1] = xptr[1] * c; [[fallthrough]];
        case 1: outptr[0] = xptr[0] * c;
        default: break;
    }
};

/**
 * @brief Performs SIMD-accelerated vector scaling (multiplication by constant) for double-precision arrays.
 *
 * Computes out[i] = x[i] * c for each element in the range [xbegin, xend) using SSE4.2 instructions.
 * Optimized for double-precision floating-point data with automatic fallback to scalar operations.
 */
inline void vecscale_sse_double(const double* xbegin,
                                const double* xend,
                                const double c,
                                std::size_t n,
                                double* out) noexcept {
    // call vecscale_sse_double function
    vecscale_sse_double(xbegin, c, n, out);
};

/**
 * @brief Performs SIMD-accelerated element-wise addition of two
 *        double-precision arrays.
 * Computes out[i] = x[i] + y[i] for each element using SSE2 instructions.
 * Optimized for 64-bit floating-point data with automatic scalar fallback.
 *
 */
inline void vecadd_sse_double(const double* x,
                              const double* y,
                              const std::size_t n,
                              const std::size_t m,
                              double* out) noexcept {
    // define ptr points to x and aligned end
    double* outptr = out;
    const double* xptr = x;
    const double* yptr = y;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // main loop
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m128d xvec0 = _mm_loadu_pd(xptr);
        __m128d xvec1 = _mm_loadu_pd(xptr + 2);
        __m128d yvec0 = _mm_loadu_pd(yptr);
        __m128d yvec1 = _mm_loadu_pd(yptr + 2);
        _mm_storeu_pd(outptr, _mm_add_pd(xvec0, yvec0));
        _mm_storeu_pd(outptr + 2, _mm_add_pd(xvec1, yvec1));
        xptr += DTYPE_UNROLLING_SIZE;
        yptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }

    // handle remaining elements
    const std::size_t remainder = end - xptr;
    switch (remainder) {
        case 3: outptr[2] = xptr[2] + yptr[2]; [[fallthrough]];
        case 2: outptr[1] = xptr[1] + yptr[1]; [[fallthrough]];
        case 1: outptr[0] = xptr[0] + yptr[0];
        default: break;
    }
};

/**
 * @brief Performs SIMD-accelerated vector addition with scaling for double-precision arrays.
 *        Computes the element-wise operation: out[i] = x[i] * c + y[i] using SSE4.2 instructions.
 */
inline void vecadd_sse_double(const double* x,
                              const double* y,
                              const double c,
                              const std::size_t n,
                              const std::size_t m,
                              double* out) noexcept {
    // define ptr points to x and aligned end
    double* outptr = out;
    const double* xptr = x;
    const double* yptr = y;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // init sum0 and sum1 to 0, load constant c into register
    __m128d sum0 = _mm_setzero_pd();
    __m128d sum1 = _mm_setzero_pd();
    const __m128d scalar = _mm_set1_pd(c);
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        // load xvec and yvec to 2 register
        __m128d xvec0 = _mm_loadu_pd(xptr);
        __m128d xvec1 = _mm_loadu_pd(xptr + 2);
        __m128d yvec0 = _mm_loadu_pd(yptr);
        __m128d yvec1 = _mm_loadu_pd(yptr + 2);
        // compute x * c + y
        sum0 = _mm_add_pd(yvec0, _mm_mul_pd(xvec0, scalar));
        sum1 = _mm_add_pd(yvec1, _mm_mul_pd(xvec1, scalar));
        // put result back to out pointer
        _mm_storeu_pd(outptr, sum0);
        _mm_storeu_pd(outptr + 2, sum1);
        // increment
        xptr += DTYPE_UNROLLING_SIZE;
        yptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }
    // handle remaining elements
    const std::size_t remainder = end - xptr;
    switch (remainder) {
        case 3: outptr[2] = xptr[2] * c + yptr[2]; [[fallthrough]];
        case 2: outptr[1] = xptr[1] * c + yptr[1]; [[fallthrough]];
        case 1: outptr[0] = xptr[0] * c + yptr[0];
        default: break;
    }
};

/**
 * @brief Performs SIMD-accelerated element-wise subtraction of two double-precision
 *        floating-point arrays. Computes out[i] = x[i] - y[i] for each element
 *        using SSE4.2 instructions.
 */
inline void vecdiff_sse_double(const double* x,
                               const double* y,
                               const std::size_t n,
                               const std::size_t m,
                               double* out) noexcept {
    // define ptr points to x and aligned end
    double* outptr = out;
    const double* xptr = x;
    const double* yptr = y;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // main SIMD loop
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        // load x, y to register
        __m128d xvec0 = _mm_loadu_pd(xptr);
        __m128d xvec1 = _mm_loadu_pd(xptr + 2);
        __m128d yvec0 = _mm_loadu_pd(yptr);
        __m128d yvec1 = _mm_loadu_pd(yptr + 2);
        // x - y
        _mm_storeu_pd(outptr, _mm_sub_pd(xvec0, yvec0));
        _mm_storeu_pd(outptr + 2, _mm_sub_pd(xvec1, yvec1));
        // increment
        xptr += DTYPE_UNROLLING_SIZE;
        yptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }

    // handle remaining elements
    const std::size_t remainder = end - xptr;
    switch (remainder) {
        case 3: outptr[2] = xptr[2] - yptr[2]; [[fallthrough]];
        case 2: outptr[1] = xptr[1] - yptr[1]; [[fallthrough]];
        case 1: outptr[0] = xptr[0] - yptr[0];
        default: break;
    }
};

/**
 * @brief Computes out[i] = c * x[i] - y[i] for each element using SSE4.2 instructions.
 */
inline void vecdiff_sse_double(const double* x,
                               const double* y,
                               const double c,
                               const std::size_t n,
                               const std::size_t m,
                               double* out) noexcept {
    // define ptr points to x and aligned end
    double* outptr = out;
    const double* xptr = x;
    const double* yptr = y;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;
    const __m128d scalar = _mm_set1_pd(c);
    // main SIMD loop
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        // load x, y to register
        __m128d xvec0 = _mm_loadu_pd(xptr);
        __m128d xvec1 = _mm_loadu_pd(xptr + 2);
        __m128d yvec0 = _mm_loadu_pd(yptr);
        __m128d yvec1 = _mm_loadu_pd(yptr + 2);
        // x - y
        _mm_storeu_pd(outptr, _mm_sub_pd(xvec0, _mm_mul_pd(yvec0, scalar)));
        _mm_storeu_pd(outptr + 2, _mm_sub_pd(xvec1, _mm_mul_pd(yvec1, scalar)));
        // increment
        xptr += DTYPE_UNROLLING_SIZE;
        yptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }

    // handle remaining elements
    const std::size_t remainder = end - xptr;
    switch (remainder) {
        case 3: outptr[2] = xptr[2] - c * yptr[2]; [[fallthrough]];
        case 2: outptr[1] = xptr[1] - c * yptr[1]; [[fallthrough]];
        case 1: outptr[0] = xptr[0] - c * yptr[0];
        default: break;
    }
};

/**
 * @brief Computes the dot product of two double-precision vectors using SSE4.2 intrinsics.
 *        Calculates the sum of element-wise products: Σ(x[i] * y[i]) for i = 0 to n-1.
 */
inline double vecdot_sse_double(const double* x,
                                const double* y,
                                std::size_t n,
                                std::size_t m) noexcept {
    const double* xptr = x;
    const double* yptr = y;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // load sum of vec to register
    __m128d sum0 = _mm_setzero_pd();
    __m128d sum1 = _mm_setzero_pd();
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        // load x, y to register
        __m128d xvec0 = _mm_loadu_pd(xptr);
        __m128d xvec1 = _mm_loadu_pd(xptr + 2);
        __m128d yvec0 = _mm_loadu_pd(yptr);
        __m128d yvec1 = _mm_loadu_pd(yptr + 2);
        // sum += x * y
        sum0 = _mm_add_pd(sum0, _mm_mul_pd(xvec0, yvec0));
        sum1 = _mm_add_pd(sum1, _mm_mul_pd(xvec1, yvec1));
        // increment
        xptr += DTYPE_UNROLLING_SIZE;
        yptr += DTYPE_UNROLLING_SIZE;
    }

    // perform a horizontal addition
    // Do not use: const __m128d sumh = _mm_hadd_pd(sums, sums);
    const __m128d combine_sum = _mm_add_pd(sum0, sum1);
    const __m128d scalar_sum = _mm_add_pd(combine_sum, _mm_shuffle_pd(combine_sum, combine_sum, 0x01));
    double total = _mm_cvtsd_f64(scalar_sum);

    const std::size_t remainder = end - xptr;
    switch (remainder) {
        case 3: total += xptr[2] * yptr[2]; [[fallthrough]];
        case 2: total += xptr[1] * yptr[1]; [[fallthrough]];
        case 1: total += xptr[0] * yptr[0];
        default: break;
    }
    return total;
};

/**
 * @brief Performs element-wise multiplication of two float arrays using SSE4.2 intrinsics.
 *        out[i] = x[i] * y[i]
 */
inline void vecmul_sse_double(const double* x,
                              const double* y,
                              std::size_t n,
                              std::size_t m,
                              double* out) noexcept {
    // define ptr points to x and aligned end
    double* outptr = out;
    const double* xptr = x;
    const double* yptr = y;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // main loop
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m128d xvec0 = _mm_loadu_pd(xptr);
        __m128d xvec1 = _mm_loadu_pd(xptr + 2);
        __m128d yvec0 = _mm_loadu_pd(yptr);
        __m128d yvec1 = _mm_loadu_pd(yptr + 2);

        _mm_storeu_pd(outptr, _mm_mul_pd(xvec0, yvec0));
        _mm_storeu_pd(outptr + 2, _mm_mul_pd(xvec1, yvec1));
        // increment
        xptr += DTYPE_UNROLLING_SIZE;
        yptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }

    // handle remaining elements
    const std::size_t remainder = end - xptr;
    switch (remainder) {
        case 3: outptr[2] = xptr[2] * yptr[2]; [[fallthrough]];
        case 2: outptr[1] = xptr[1] * yptr[1]; [[fallthrough]];
        case 1: outptr[0] = xptr[0] * yptr[0];
        default: break;
    }
};

/**
 * @brief Computes the accumulated sum of double-precision elements
 *        using SSE4.2 intrinsics
 */
inline double vecaccmul_sse_double(const double* xbegin,
                                   const double* xend,
                                   std::size_t n) noexcept {
    const double* xptr = xbegin;
    const double* end = xbegin + n;
    double total = 0.0;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // init sum0 and sum1 to 0
    __m128d sum0 = _mm_setzero_pd();
    __m128d sum1 = _mm_setzero_pd();
    // main loop
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        // load 4 elements from x to 2 register
        __m128d xvec0 = _mm_loadu_pd(xptr);
        __m128d xvec1 = _mm_loadu_pd(xptr + 2);
        // compute sum = sum + vec * vec
        sum0 = _mm_add_pd(sum0, xvec0);
        sum1 = _mm_add_pd(sum1, xvec1);
        xptr += DTYPE_UNROLLING_SIZE;
    }

    // perform a horizontal addition
    const __m128d combine_sum = _mm_add_pd(sum0, sum1);
    const __m128d scalar_sum = _mm_add_pd(combine_sum, _mm_shuffle_pd(combine_sum, combine_sum, 0x01));
    total += _mm_cvtsd_f64(scalar_sum);

    // handle remaining elements
    const std::size_t remainder = end - xptr;
    switch (remainder) {
        case 3: total += xptr[2]; [[fallthrough]];
        case 2: total += xptr[1]; [[fallthrough]];
        case 1: total += xptr[0];
        default: break;
    }
    return total;
};

#endif

}
}
#endif // MATH_KERNELS_OPS_SSE_OPS_SSE_DOUBLE_HPP_
