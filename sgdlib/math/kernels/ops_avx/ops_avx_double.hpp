#ifndef MATH_KERNELS_OPS_AVX_OPS_AVX_DOUBLE_HPP_
#define MATH_KERNELS_OPS_AVX_OPS_AVX_DOUBLE_HPP_

#include "common/constants.hpp"
#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

#if defined(USE_AVX)

/**
 * @brief Sets all elements of a double array to a constant value using AVX instructions.
 */
inline void vecset_avx_double(double* x, const double c, std::size_t n) noexcept {
    if (x == nullptr || n == 0) return;

    // handle small size n < 8
    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = c;
        }
        return;
    }

    // define a ptr points to x, end of bound
    double* xptr = x;
    const double* end = x + n;

    // load c into register
    const __m256d scalar = _mm256_set1_pd(c);

    // main loop
    for (; xptr + DTYPE_UNROLLING_SIZE <= end ; xptr += DTYPE_UNROLLING_SIZE) {
        _mm256_storeu_pd(xptr, scalar);
        _mm256_storeu_pd(xptr + 4, scalar);
        _mm256_storeu_pd(xptr + 8, scalar);
        _mm256_storeu_pd(xptr + 12, scalar);
    }

    // handle remaining elements
    const std::size_t remainder = end - xptr;
    if (remainder >= DTYPE_ELEMS_PER_REGISTER) {
        const std::size_t num_blocks = remainder / 4;
        for (std::size_t i = 0; i < num_blocks; ++i) {
            _mm256_storeu_pd(xptr, scalar);
            xptr += 4;
        }
    }

    // handle remaining elements
    const std::size_t tails = end - xptr;
    switch (tails) {
        case 3: xptr[2] = c; [[fallthrough]];
        case 2: xptr[1] = c; [[fallthrough]];
        case 1: xptr[0] = c;
        default: break;
    }
}

/**
 * @brief Copies an array of double floating-point numbers using AVX2 instructions.
 */
void veccpy_avx_double(const double* x, std::size_t n, double* out) noexcept {
    if (n == 0) return ;
    if (x == nullptr) return ;
    if (out == nullptr) return ;
    // handle small size n < 4
    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i];
        }
        return ;
    }

    // define xptr, outptr and a ptr to end of x
    double* outptr = out;
    const double* xptr = x;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // main loop to process simd
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256d xvec0 = _mm256_loadu_pd(xptr);
        __m256d xvec1 = _mm256_loadu_pd(xptr + 4);
        __m256d xvec2 = _mm256_loadu_pd(xptr + 8);
        __m256d xvec3 = _mm256_loadu_pd(xptr + 12);
        _mm256_storeu_pd(outptr, xvec0);
        _mm256_storeu_pd(outptr + 4, xvec1);
        _mm256_storeu_pd(outptr + 8, xvec2);
        _mm256_storeu_pd(outptr + 12, xvec3);
        // increment
        xptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }

    // handle the last 4 - 15 remaining elements
    const std::size_t remainder = end - xptr;
    if (remainder >= DTYPE_ELEMS_PER_REGISTER) {
        const std::size_t num_blocks = remainder / DTYPE_ELEMS_PER_REGISTER;
        for (std::size_t i = 0; i < num_blocks; ++i) {
            const __m256d xvec = _mm256_loadu_pd(xptr);
            _mm256_storeu_pd(outptr, xvec);
            xptr += DTYPE_ELEMS_PER_REGISTER;
            outptr += DTYPE_ELEMS_PER_REGISTER;
        }
    }

    // handle remaining elements
    const std::size_t tails = end - xptr;
    switch (tails) {
        case 3: outptr[2] = xptr[2]; [[fallthrough]];
        case 2: outptr[1] = xptr[1]; [[fallthrough]];
        case 1: outptr[0] = xptr[0];
        default: break;
    }
};

/**
 * @brief Copies elements from the input array `x` to the output array `out`and
 *        negates their values using AVX instructions.
 */
void vecncpy_avx_double(const double* x, std::size_t n, double* out) noexcept {
    if (n == 0) return ;
    if (x == nullptr) return ;
    if (out == nullptr) return ;
    // handle small size n < 4
    if (n < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = -x[i];
        }
        return ;
    }

    // define xptr, outptr and a ptr to end of x
    double* outptr = out;
    const double* xptr = x;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // define sign_flip mask
    // const double* aligned_end = xptr + ((end - xptr) & ~3ULL);
    const __m256d sign_flip = _mm256_castsi256_pd(
        _mm256_set1_epi64x(0x8000000000000000)
    );

    // main loop to process simd
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256d xvec0 = _mm256_loadu_pd(xptr);
        __m256d xvec1 = _mm256_loadu_pd(xptr + 4);
        __m256d xvec2 = _mm256_loadu_pd(xptr + 8);
        __m256d xvec3 = _mm256_loadu_pd(xptr + 12);

        xvec0 = _mm256_xor_pd(xvec0, sign_flip);
        xvec1 = _mm256_xor_pd(xvec1, sign_flip);
        xvec2 = _mm256_xor_pd(xvec2, sign_flip);
        xvec3 = _mm256_xor_pd(xvec3, sign_flip);

        _mm256_storeu_pd(outptr, xvec0);
        _mm256_storeu_pd(outptr + 4, xvec1);
        _mm256_storeu_pd(outptr + 8, xvec2);
        _mm256_storeu_pd(outptr + 12, xvec3);
        // increment
        xptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }

    // handle the last 4 - 15 remaining elements
    const std::size_t remainder = end - xptr;
    if (remainder >= DTYPE_ELEMS_PER_REGISTER) {
        const std::size_t num_blocks = remainder / DTYPE_ELEMS_PER_REGISTER;
        for (std::size_t i = 0; i < num_blocks; ++i) {
            __m256d xvec = _mm256_loadu_pd(xptr);
            xvec = _mm256_xor_pd(xvec, sign_flip);
            _mm256_storeu_pd(outptr, xvec);
            xptr += DTYPE_ELEMS_PER_REGISTER;
            outptr += DTYPE_ELEMS_PER_REGISTER;
        }
    }

    // handle remaining 0 - 3 elements
    const std::size_t tails = end - xptr;
    switch (tails) {
        case 3: outptr[2] = -xptr[2]; [[fallthrough]];
        case 2: outptr[1] = -xptr[1]; [[fallthrough]];
        case 1: outptr[0] = -xptr[0];
        default: break;
    }
};

/**
 * @brief Clips elements of a double array to specified range using AVX intrinsics
 */
inline void vecclip_avx_double(double* x, double min, double max, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return ;
    if (min > max) return ;

    // check if x is aligned to 16 bytes == 2 elems
    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = std::clamp(x[i], min, max);
            // x[i] = (x[i] < min) ? min : (x[i] > max) ? max : x[i];
        }
        return ;
    }

    // define ptr points to x and end of x
    double* xptr = x;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // Create an SSE register with all elements set to the min/max value
    const __m256d xmin = _mm256_set1_pd(min);
    const __m256d xmax = _mm256_set1_pd(max);

    // Process the array in chunks of 2 elements using SSE intrinsics
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256d xvec0 = _mm256_loadu_pd(xptr);
        __m256d xvec1 = _mm256_loadu_pd(xptr + 4);
        __m256d xvec2 = _mm256_loadu_pd(xptr + 8);
        __m256d xvec3 = _mm256_loadu_pd(xptr + 12);
        xvec0 = _mm256_min_pd(_mm256_max_pd(xvec0, xmin), xmax);
        xvec1 = _mm256_min_pd(_mm256_max_pd(xvec1, xmin), xmax);
        xvec2 = _mm256_min_pd(_mm256_max_pd(xvec2, xmin), xmax);
        xvec3 = _mm256_min_pd(_mm256_max_pd(xvec3, xmin), xmax);
        _mm256_storeu_pd(xptr, xvec0);
        _mm256_storeu_pd(xptr + 4, xvec1);
        _mm256_storeu_pd(xptr + 8, xvec2);
        _mm256_storeu_pd(xptr + 12, xvec3);
        // increment
        xptr += DTYPE_UNROLLING_SIZE;
    }

    // handle the last 4 - 15 remaining elements
    const std::size_t remainder = end - xptr;
    if (remainder >= DTYPE_ELEMS_PER_REGISTER) {
        const std::size_t num_blocks = remainder / DTYPE_ELEMS_PER_REGISTER;
        for (std::size_t i = 0; i < num_blocks; ++i) {
            __m256d xvec = _mm256_loadu_pd(xptr);
            xvec = _mm256_min_pd(_mm256_max_pd(xvec, xmin), xmax);
            _mm256_storeu_pd(xptr, xvec);
            xptr += DTYPE_ELEMS_PER_REGISTER;
        }
    }

    // handle remaining 0 - 3 elements
    const std::size_t tails = end - xptr;
    switch (tails) {
        case 3: xptr[2] = (xptr[2] < min) ? min : (xptr[2] > max) ? max : xptr[2]; [[fallthrough]];
        case 2: xptr[1] = (xptr[1] < min) ? min : (xptr[1] > max) ? max : xptr[1]; [[fallthrough]];
        case 1: xptr[0] = (xptr[0] < min) ? min : (xptr[0] > max) ? max : xptr[0];
        default: break;
    }
};

/**
 * @brief Checks for infinite values in a double array using AVX intrinsics
 */
inline bool hasinf_avx_double(const double* x, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return false;

    // check if x has 2 elems
    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isinf(x[i])) {
                return true;
            }
        }
        return false;
    }

    // define ptr points to x and end of x
    const double* xptr = x;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // load x positif inf + x negative inf to SIMD register
    const __m256d pos_inf = _mm256_set1_pd(INF);
    const __m256d neg_inf = _mm256_set1_pd(-INF);

    // loop the array in chunks of 4 elements
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256d xvec0 = _mm256_loadu_pd(xptr);
        __m256d xvec1 = _mm256_loadu_pd(xptr + 4);
        __m256d xvec2 = _mm256_loadu_pd(xptr + 8);
        __m256d xvec3 = _mm256_loadu_pd(xptr + 12);
        __m256d cmp0 = _mm256_or_pd(_mm256_cmp_pd(xvec0, pos_inf, _CMP_EQ_OQ),
                                    _mm256_cmp_pd(xvec0, neg_inf, _CMP_EQ_OQ));
        __m256d cmp1 = _mm256_or_pd(_mm256_cmp_pd(xvec1, pos_inf, _CMP_EQ_OQ),
                                    _mm256_cmp_pd(xvec1, neg_inf, _CMP_EQ_OQ));
        __m256d cmp2 = _mm256_or_pd(_mm256_cmp_pd(xvec2, pos_inf, _CMP_EQ_OQ),
                                    _mm256_cmp_pd(xvec2, neg_inf, _CMP_EQ_OQ));
        __m256d cmp3 = _mm256_or_pd(_mm256_cmp_pd(xvec3, pos_inf, _CMP_EQ_OQ),
                                    _mm256_cmp_pd(xvec3, neg_inf, _CMP_EQ_OQ));
        if (_mm256_movemask_pd(_mm256_or_pd(
                               _mm256_or_pd(cmp0, cmp1),
                               _mm256_or_pd(cmp2, cmp3))) != 0) {
            return true;
        }
        xptr += DTYPE_UNROLLING_SIZE;
    }

    // handle the last 4 - 15 remaining elements
    const std::size_t remainder = end - xptr;
    if (remainder >= DTYPE_ELEMS_PER_REGISTER) {
        const std::size_t num_blocks = remainder / DTYPE_ELEMS_PER_REGISTER;
        for (std::size_t i = 0; i < num_blocks; ++i) {
            const __m256d xvec = _mm256_loadu_pd(xptr);
            const __m256d cmp = _mm256_or_pd(_mm256_cmp_pd(xvec, pos_inf, _CMP_EQ_OQ),
                                             _mm256_cmp_pd(xvec, neg_inf, _CMP_EQ_OQ));
            if (_mm256_movemask_pd(cmp) != 0) {
                return true;
            }
            xptr += DTYPE_ELEMS_PER_REGISTER;
        }
    }

    // process remaining elements
    if (end > xptr) {
        const std::size_t tails = end - xptr;
        for (std::size_t i = 0; i < tails; ++i) {
            if (std::isinf(xptr[i])) {
                return true;
            }
        }
    }
    return false;
};

/**
 * @brief Computes the squared L2 norm (Euclidean norm) of a double-precision vector using AVX intrinsics.
 */
inline double vecnorm2_avx_double(const double* x,
                                  std::size_t n,
                                  bool squared) noexcept {
    if (n == 0 || x == nullptr) return 0.0;
    // For small arrays with less than or equal to two elements
    if (n < 16) {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            sum += x[i] * x[i];
        }
        return squared ? sum : std::sqrt(sum);
    }

    // define xptr and end point to x and end of x
    const double* xptr = x;
    const double* end = x + n;
    double total = 0.0;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // init sum to 0
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d sum3 = _mm256_setzero_pd();
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        // load 16 elements from x to 4 SIMD register
        __m256d xvec0 = _mm256_loadu_pd(xptr);
        __m256d xvec1 = _mm256_loadu_pd(xptr + 4);
        __m256d xvec2 = _mm256_loadu_pd(xptr + 8);
        __m256d xvec3 = _mm256_loadu_pd(xptr + 12);
        // compute sum = sum + vec * vec
        // sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(xvec0, xvec0));
        // sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(xvec1, xvec1));
        // sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(xvec2, xvec2));
        // sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(xvec3, xvec3));
        sum0 = _mm256_fmadd_pd(xvec0, xvec0, sum0);
        sum1 = _mm256_fmadd_pd(xvec1, xvec1, sum1);
        sum2 = _mm256_fmadd_pd(xvec2, xvec2, sum2);
        sum3 = _mm256_fmadd_pd(xvec3, xvec3, sum3);
        // increment
        xptr += DTYPE_UNROLLING_SIZE;
    }
    // combine sum0, sum1, sum2 and sum3
    __m256d partial_sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1),
                                        _mm256_add_pd(sum2, sum3));

    // handle teh last 4 - 15 elements
    const std::size_t remainder = end - xptr;
    if (remainder >= DTYPE_ELEMS_PER_REGISTER) {
        const std::size_t num_blocks = remainder / DTYPE_ELEMS_PER_REGISTER;
        for (std::size_t i = 0; i < num_blocks; ++i) {
            const __m256d xvec = _mm256_loadu_pd(xptr);
            // sums = _mm256_add_pd(sums, _mm256_mul_pd(xvec, xvec));
            partial_sum = _mm256_fmadd_pd(xvec, xvec, partial_sum);
            xptr += DTYPE_ELEMS_PER_REGISTER;
        }
    }

    // perform a horizontal addition
    // [a,b,c,d] -> [c,d,a,b]
    // [a+c,b+d,c+a,d+d]
    // [(a+c)+(b+d),*,*,*]
    const __m256d perm = _mm256_permute2f128_pd(partial_sum, partial_sum, 0x01);
    const __m256d combine_sum = _mm256_add_pd(partial_sum, perm);
    const __m256d scalar_sum = _mm256_hadd_pd(combine_sum, combine_sum);
    total += _mm256_cvtsd_f64(scalar_sum);
    // sums = _mm256_add_pd(sums, _mm256_permute2f128_pd(sums, sums, 0x01));
    // sums = _mm256_hadd_pd(sums, sums);
    // total += _mm256_cvtsd_f64(sums);

    // process remaining elements
    const std::size_t tails = end - xptr;
    switch (tails) {
        case 3: total += xptr[2] * xptr[2]; [[fallthrough]];
        case 2: total += xptr[1] * xptr[1]; [[fallthrough]];
        case 1: total += xptr[0] * xptr[0];
        default: break;
    }
    return squared ? total : std::sqrt(total);
};

/**
 * @brief Computes the squared L2 norm of a double-precision vector using AVX2 intrinsics.
 */
inline double vecnorm1_avx_double(const double* x,
                                  std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return 0.0;
    // For small arrays with less than or equal to two elements
    if (n < 16) {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            sum += std::abs(x[i]);
        }
        return sum;
    }

    // define ptr points to x and end of x
    const double* xptr = x;
    const double* end = x + n;
    double total = 0.0;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // mask for the absolute value of the a double
    const __m256d abs_mask = _mm256_castsi256_pd(
        _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));

    // init sum to 0
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d sum3 = _mm256_setzero_pd();
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        // load 16 elements from x to 4 SIMD register
        __m256d xvec0 = _mm256_loadu_pd(xptr);
        __m256d xvec1 = _mm256_loadu_pd(xptr + 4);
        __m256d xvec2 = _mm256_loadu_pd(xptr + 8);
        __m256d xvec3 = _mm256_loadu_pd(xptr + 12);
        sum0 = _mm256_add_pd(sum0, _mm256_and_pd(xvec0, abs_mask));
        sum1 = _mm256_add_pd(sum1, _mm256_and_pd(xvec1, abs_mask));
        sum2 = _mm256_add_pd(sum2, _mm256_and_pd(xvec2, abs_mask));
        sum3 = _mm256_add_pd(sum3, _mm256_and_pd(xvec3, abs_mask));
        xptr += DTYPE_UNROLLING_SIZE;
    }
    // combine sum0, sum1, sum2 and sum3
    __m256d partial_sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1),
                                        _mm256_add_pd(sum2, sum3));

    // handle teh last 4 - 15 elements
    const std::size_t remainder = end - xptr;
    if (remainder >= DTYPE_ELEMS_PER_REGISTER) {
        const std::size_t num_blocks = remainder / DTYPE_ELEMS_PER_REGISTER;
        for (std::size_t i = 0; i < num_blocks; ++i) {
            const __m256d xvec = _mm256_loadu_pd(xptr);
            partial_sum = _mm256_add_pd(partial_sum, _mm256_and_pd(xvec, abs_mask));
            xptr += DTYPE_ELEMS_PER_REGISTER;
        }
    }

    // perform a horizontal addition of the 16 channels
    // [a,b,c,d] -> [c,d,a,b]
    // [a+c,b+d,c+a,d+b]
    // [(a+c)+(b+d),*,*,*]
    const __m256d perm = _mm256_permute2f128_pd(partial_sum, partial_sum, 0x01);
    const __m256d combine_sum = _mm256_add_pd(partial_sum, perm);
    const __m256d scalar_sum = _mm256_hadd_pd(combine_sum, combine_sum);
    total += _mm256_cvtsd_f64(scalar_sum);
    // sums = _mm256_add_pd(sums, _mm256_permute2f128_pd(sums, sums, 0x01));
    // sums = _mm256_hadd_pd(sums, sums);
    // total += _mm256_cvtsd_f64(sums);

    // handle remaining elements
    const std::size_t tails = end - xptr;
    switch (tails) {
        case 3: total += std::abs(xptr[2]); [[fallthrough]];
        case 2: total += std::abs(xptr[1]); [[fallthrough]];
        case 1: total += std::abs(xptr[0]);
        default: break;
    }
    return total;
};

/**
 * @brief Scales a double array in-place by constant factor using AVX vectorization.
 */
inline void vecscale_avx_double(const double* x,
                                const double c,
                                std::size_t n,
                                double* out) noexcept {
    if (x == nullptr || out == nullptr) return ;
    if (n == 0 || c == 1.0) return ;

    // for small size n < 4
    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i] * c;
        }
        return ;
    }

     // define ptr points to x and end of x
     double* outptr = out;
     const double* xptr = x;
     const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // load constant c into register
    const __m256d scalar = _mm256_set1_pd(c);
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256d xvec0 = _mm256_loadu_pd(xptr);
        __m256d xvec1 = _mm256_loadu_pd(xptr + 4);
        __m256d xvec2 = _mm256_loadu_pd(xptr + 8);
        __m256d xvec3 = _mm256_loadu_pd(xptr + 12);
        _mm256_storeu_pd(outptr, _mm256_mul_pd(xvec0, scalar));
        _mm256_storeu_pd(outptr + 4, _mm256_mul_pd(xvec1, scalar));
        _mm256_storeu_pd(outptr + 8, _mm256_mul_pd(xvec2, scalar));
        _mm256_storeu_pd(outptr + 12, _mm256_mul_pd(xvec3, scalar));
        xptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }

    // handle the last 4 - 15 elements
    const std::size_t remainder = end - xptr;
    if (remainder >= DTYPE_ELEMS_PER_REGISTER) {
        const std::size_t num_blocks = remainder / DTYPE_ELEMS_PER_REGISTER;
        for (std::size_t i = 0; i < num_blocks; ++i) {
            __m256d xvec = _mm256_loadu_pd(xptr);
            _mm256_storeu_pd(outptr, _mm256_mul_pd(xvec, scalar));
            xptr += DTYPE_ELEMS_PER_REGISTER;
            outptr += DTYPE_ELEMS_PER_REGISTER;
        }
    }

    // handle lats 1 - 3 remaining elements
    const std::size_t tails = end - xptr;
    switch (tails) {
        case 3: outptr[2] = xptr[2] * c; [[fallthrough]];
        case 2: outptr[1] = xptr[1] * c; [[fallthrough]];
        case 1: outptr[0] = xptr[0] * c;
        default: break;
    }
};

/**
 * @brief Scales a double array using AVX vectorization and stores result in output buffer
 *        out[i] = c * x[i] + y[i]
 */
inline void vecscale_avx_double(const double* xbegin,
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
    vecscale_avx_double(xbegin, c, n, out);
};

/**
 *
 */
inline void vecadd_avx_double(const double* x,
                              const double* y,
                              const std::size_t n,
                              const std::size_t m,
                              double* out) noexcept {
    if (x == nullptr || y == nullptr) return ;
    if (out == nullptr) return ;
    if (n == 0 || m == 0) return ;
    if (m != n) return ;

    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i] + y[i];
        }
        return ;
    }

    // define ptr points to x and aligned end
    double* outptr = out;
    const double* xptr = x;
    const double* yptr = y;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // main SIMD loop
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256d xvec0 = _mm256_loadu_pd(xptr);
        __m256d yvec0 = _mm256_loadu_pd(yptr);
        __m256d xvec1 = _mm256_loadu_pd(xptr + 4);
        __m256d yvec1 = _mm256_loadu_pd(yptr + 4);
        __m256d xvec2 = _mm256_loadu_pd(xptr + 8);
        __m256d yvec2 = _mm256_loadu_pd(yptr + 8);
        __m256d xvec3 = _mm256_loadu_pd(xptr + 12);
        __m256d yvec3 = _mm256_loadu_pd(yptr + 12);
        _mm256_storeu_pd(outptr, _mm256_add_pd(xvec0, yvec0));
        _mm256_storeu_pd(outptr + 4, _mm256_add_pd(xvec1, yvec1));
        _mm256_storeu_pd(outptr + 8, _mm256_add_pd(xvec2, yvec2));
        _mm256_storeu_pd(outptr + 12, _mm256_add_pd(xvec3, yvec3));
        xptr += DTYPE_UNROLLING_SIZE;
        yptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }

    // handle the last 4 - 15 elements
    const std::size_t remainder = end - xptr;
    if (remainder >= DTYPE_ELEMS_PER_REGISTER) {
        const std::size_t num_blocks = remainder / DTYPE_ELEMS_PER_REGISTER;
        for (std::size_t i = 0; i < num_blocks; ++i) {
            __m256d xvec = _mm256_loadu_pd(xptr);
            __m256d yvec = _mm256_loadu_pd(yptr);
            _mm256_storeu_pd(outptr, _mm256_add_pd(xvec, yvec));
            xptr += DTYPE_ELEMS_PER_REGISTER;
            yptr += DTYPE_ELEMS_PER_REGISTER;
            outptr += DTYPE_ELEMS_PER_REGISTER;
        }
    }

    // handle remaining elements
    const std::size_t tails = end - xptr;
    switch (tails) {
        case 3: outptr[2] = xptr[2] + yptr[2]; [[fallthrough]];
        case 2: outptr[1] = xptr[1] + yptr[1]; [[fallthrough]];
        case 1: outptr[0] = xptr[0] + yptr[0];
        default: break;
    }
};

/**
 * @brief Performs vector addition with scaling: out = c * x + y
 */
inline void vecadd_avx_double(const double* x,
                              const double* y,
                              const double c,
                              const std::size_t n,
                              const std::size_t m,
                              double* out) noexcept {
    if (x == nullptr || y == nullptr) return ;
    if (out == nullptr) return ;
    if (n == 0 || m == 0) return ;
    if (m != n) return ;

    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = c * x[i] + y[i];
        }
        return ;
    }

    // define ptr points to x and aligned end
    double* outptr = out;
    const double* xptr = x;
    const double* yptr = y;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // load constant c into register
    const __m256d scalar = _mm256_set1_pd(c);
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256d xvec0 = _mm256_loadu_pd(xptr);
        __m256d yvec0 = _mm256_loadu_pd(yptr);
        __m256d xvec1 = _mm256_loadu_pd(xptr + 4);
        __m256d yvec1 = _mm256_loadu_pd(yptr + 4);
        __m256d xvec2 = _mm256_loadu_pd(xptr + 8);
        __m256d yvec2 = _mm256_loadu_pd(yptr + 8);
        __m256d xvec3 = _mm256_loadu_pd(xptr + 12);
        __m256d yvec3 = _mm256_loadu_pd(yptr + 12);
        // _mm256_storeu_pd(outptr, _mm256_add_pd(_mm256_mul_pd(xvec, scalar), yvec));
        _mm256_storeu_pd(outptr, _mm256_fmadd_pd(xvec0, scalar, yvec0));
        _mm256_storeu_pd(outptr + 4, _mm256_fmadd_pd(xvec1, scalar, yvec1));
        _mm256_storeu_pd(outptr + 8, _mm256_fmadd_pd(xvec2, scalar, yvec2));
        _mm256_storeu_pd(outptr + 12, _mm256_fmadd_pd(xvec3, scalar, yvec3));
        xptr += DTYPE_UNROLLING_SIZE;
        yptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }

    // handle the last 4 - 15 elements
    const std::size_t remainder = end - xptr;
    if (remainder >= DTYPE_ELEMS_PER_REGISTER) {
        const std::size_t num_blocks = remainder / DTYPE_ELEMS_PER_REGISTER;
        for (std::size_t i = 0; i < num_blocks; ++i) {
            __m256d xvec = _mm256_loadu_pd(xptr);
            __m256d yvec = _mm256_loadu_pd(yptr);
            // _mm256_storeu_pd(outptr, _mm256_add_pd(xvec, yvec));
            _mm256_storeu_pd(outptr, _mm256_fmadd_pd(xvec, scalar, yvec));
            xptr += DTYPE_ELEMS_PER_REGISTER;
            yptr += DTYPE_ELEMS_PER_REGISTER;
            outptr += DTYPE_ELEMS_PER_REGISTER;
        }
    }

    // handle remaining elements
    const std::size_t tails = end - xptr;
    switch (tails) {
        case 3: outptr[2] = c * xptr[2] + yptr[2]; [[fallthrough]];
        case 2: outptr[1] = c * xptr[1] + yptr[1]; [[fallthrough]];
        case 1: outptr[0] = c * xptr[0] + yptr[0];
        default: break;
    }
};

/**
 * @brief Computes element-wise difference between two double arrays using AVX vectorization
 */
inline void vecdiff_avx_double(const double* x,
                               const double* y,
                               const std::size_t n,
                               const std::size_t m,
                               double* out) noexcept {
    if (x == nullptr || y == nullptr) return ;
    if (out == nullptr) return ;
    if (n == 0 || m == 0) return ;
    if (m != n) return ;

    if (n < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i] - y[i];
        }
        return ;
    }

    // define ptr points to x and aligned end
    double* outptr = out;
    const double* xptr = x;
    const double* yptr = y;
    const double* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256d xvec0 = _mm256_loadu_pd(xptr);
        __m256d yvec0 = _mm256_loadu_pd(yptr);
        __m256d xvec1 = _mm256_loadu_pd(xptr + 4);
        __m256d yvec1 = _mm256_loadu_pd(yptr + 4);
        __m256d xvec2 = _mm256_loadu_pd(xptr + 8);
        __m256d yvec2 = _mm256_loadu_pd(yptr + 8);
        __m256d xvec3 = _mm256_loadu_pd(xptr + 12);
        __m256d yvec3 = _mm256_loadu_pd(yptr + 12);
        _mm256_storeu_pd(outptr, _mm256_sub_pd(xvec0, yvec0));
        _mm256_storeu_pd(outptr + 4, _mm256_sub_pd(xvec1, yvec1));
        _mm256_storeu_pd(outptr + 8, _mm256_sub_pd(xvec2, yvec2));
        _mm256_storeu_pd(outptr + 12, _mm256_sub_pd(xvec3, yvec3));
        xptr += DTYPE_UNROLLING_SIZE;
        yptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }

    // handle the last 4 - 15 elements
    const std::size_t remainder = end - xptr;
    if (remainder >= DTYPE_ELEMS_PER_REGISTER) {
        const std::size_t num_blocks = remainder / DTYPE_ELEMS_PER_REGISTER;
        for (std::size_t i = 0; i < num_blocks; ++i) {
            __m256d xvec = _mm256_loadu_pd(xptr);
            __m256d yvec = _mm256_loadu_pd(yptr);
            _mm256_storeu_pd(outptr, _mm256_sub_pd(xvec, yvec));
            xptr += DTYPE_ELEMS_PER_REGISTER;
            yptr += DTYPE_ELEMS_PER_REGISTER;
            outptr += DTYPE_ELEMS_PER_REGISTER;
        }
    }

    // handle remaining elements
    const std::size_t tails = end - xptr;
    switch (tails) {
        case 3: outptr[2] = xptr[2] - yptr[2]; [[fallthrough]];
        case 2: outptr[1] = xptr[1] - yptr[1]; [[fallthrough]];
        case 1: outptr[0] = xptr[0] - yptr[0];
        default: break;
    }
};

/**
 * @brief Computes the dot product of two double arrays using AVX vectorization
 */
inline double vecdot_avx_double(const double* x,
                                const double* y,
                                std::size_t n,
                                std::size_t m) noexcept {
    if (x == nullptr || y == nullptr) return 0.0;
    if (n == 0 || m == 0) return 0.0;
    if (n != m) return 0.0;

    // for small size array
    if (n < 16) {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    const double* xptr = x;
    const double* yptr = y;
    const double* end = x + n;
    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;
    // load sum of vec to register
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d sum3 = _mm256_setzero_pd();
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256d xvec0 = _mm256_loadu_pd(xptr);
        __m256d yvec0 = _mm256_loadu_pd(yptr);
        __m256d xvec1 = _mm256_loadu_pd(xptr + 4);
        __m256d yvec1 = _mm256_loadu_pd(yptr + 4);
        __m256d xvec2 = _mm256_loadu_pd(xptr + 8);
        __m256d yvec2 = _mm256_loadu_pd(yptr + 8);
        __m256d xvec3 = _mm256_loadu_pd(xptr + 12);
        __m256d yvec3 = _mm256_loadu_pd(yptr + 12);
        // sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(xvec0, yvec0));
        sum0 = _mm256_fmadd_pd(xvec0, yvec0, sum0);
        sum1 = _mm256_fmadd_pd(xvec1, yvec1, sum1);
        sum2 = _mm256_fmadd_pd(xvec2, yvec2, sum2);
        sum3 = _mm256_fmadd_pd(xvec3, yvec3, sum3);
        xptr += DTYPE_UNROLLING_SIZE;
        yptr += DTYPE_UNROLLING_SIZE;
    }

    // combine sum0, sum1, sum2 and sum3
    __m256d partial_sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1),
                                        _mm256_add_pd(sum2, sum3));

    // handle the last 4 - 15 elements
    const std::size_t remainder = end - xptr;
    if (remainder >= DTYPE_ELEMS_PER_REGISTER) {
        const std::size_t num_blocks = remainder / DTYPE_ELEMS_PER_REGISTER;
        for (std::size_t i = 0; i < num_blocks; ++i) {
            __m256d xvec = _mm256_loadu_pd(xptr);
            __m256d yvec = _mm256_loadu_pd(yptr);
            partial_sum = _mm256_fmadd_pd(xvec, yvec, partial_sum);
            xptr += DTYPE_ELEMS_PER_REGISTER;
            yptr += DTYPE_ELEMS_PER_REGISTER;
        }
    }

    // perform a horizontal addition
    const __m256d perm = _mm256_permute2f128_pd(partial_sum, partial_sum, 0x01);
    const __m256d combine_sum = _mm256_add_pd(partial_sum, perm);
    const __m256d scalar_sum = _mm256_hadd_pd(combine_sum, combine_sum);
    double total = _mm256_cvtsd_f64(scalar_sum);

    // handle remaining elements
    const std::size_t tails = end - xptr;
    switch (tails) {
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
inline void vecmul_avx_double(const double* x,
                              const double* y,
                              std::size_t n,
                              std::size_t m,
                              double* out) noexcept {
    if (x == nullptr || y == nullptr) return ;
    if (out == nullptr) return ;
    if (n == 0 || m == 0) return ;
    if (m != n) return ;

    // handle small size case n < 4
    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i] * y[i];
        }
        return ;
    }

    // define pointers
    double* outptr = out;
    const double* xptr = x;
    const double* yptr = y;
    const double* end = x + n;

    // define unrolling param
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256d xvec0 = _mm256_loadu_pd(xptr);
        __m256d yvec0 = _mm256_loadu_pd(yptr);
        __m256d xvec1 = _mm256_loadu_pd(xptr + 4);
        __m256d yvec1 = _mm256_loadu_pd(yptr + 4);
        __m256d xvec2 = _mm256_loadu_pd(xptr + 8);
        __m256d yvec2 = _mm256_loadu_pd(yptr + 8);
        __m256d xvec3 = _mm256_loadu_pd(xptr + 12);
        __m256d yvec3 = _mm256_loadu_pd(yptr + 12);
        _mm256_storeu_pd(outptr, _mm256_mul_pd(xvec0, yvec0));
        _mm256_storeu_pd(outptr + 4, _mm256_mul_pd(xvec1, yvec1));
        _mm256_storeu_pd(outptr + 8, _mm256_mul_pd(xvec2, yvec2));
        _mm256_storeu_pd(outptr + 12, _mm256_mul_pd(xvec3, yvec3));
        xptr += DTYPE_UNROLLING_SIZE;
        yptr += DTYPE_UNROLLING_SIZE;
        outptr += DTYPE_UNROLLING_SIZE;
    }

    // handle the last 4 - 15 elements
    const std::size_t remainder = end - xptr;
    if (remainder >= DTYPE_ELEMS_PER_REGISTER) {
        const std::size_t num_blocks = remainder / DTYPE_ELEMS_PER_REGISTER;
        for (std::size_t i = 0; i < num_blocks; ++i) {
            __m256d xvec = _mm256_loadu_pd(xptr);
            __m256d yvec = _mm256_loadu_pd(yptr);
            _mm256_storeu_pd(outptr, _mm256_mul_pd(xvec, yvec));
            xptr += DTYPE_ELEMS_PER_REGISTER;
            yptr += DTYPE_ELEMS_PER_REGISTER;
            outptr += DTYPE_ELEMS_PER_REGISTER;
        }
    }

    // handle remaining elements
    const std::size_t tails = end - xptr;
    switch (tails) {
        case 3: outptr[2] = xptr[2] * yptr[2]; [[fallthrough]];
        case 2: outptr[1] = xptr[1] * yptr[1]; [[fallthrough]];
        case 1: outptr[0] = xptr[0] * yptr[0];
        default: break;
    }
};

/**
 * @brief Computes the accumulated sum of double-precision elements
 *        using AVX2 intrinsics
 */
inline double vecaccmul_sse_double(const double* xbegin,
                                   const double* xend,
                                   std::size_t n) noexcept {
    if (xbegin == nullptr || xend == nullptr) return 0.0;
    if (xend <= xbegin) return 0.0;
    const std::size_t m = static_cast<std::size_t>(xend - xbegin);
    if (n != m) return 0.0;
    if (n == 0) return 0.0;
    if (n < 16) {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            sum += x[i];
        }
        return sum;
    }

    // define xptr and end point to x and end of x
    const double* xptr = x;
    const double* end = x + n;
    double total = 0.0;

    // loop unrolling
    const std::size_t num_unrolls = n / DTYPE_UNROLLING_SIZE;

    // init sum to 0
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d sum3 = _mm256_setzero_pd();
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        // load 16 elements from x to 4 SIMD register
        __m256d xvec0 = _mm256_loadu_pd(xptr);
        __m256d xvec1 = _mm256_loadu_pd(xptr + 4);
        __m256d xvec2 = _mm256_loadu_pd(xptr + 8);
        __m256d xvec3 = _mm256_loadu_pd(xptr + 12);
        // compute sum = sum + vec
        sum0 = _mm256_add_pd(sum0, xvec0);
        sum1 = _mm256_add_pd(sum1, xvec1);
        sum2 = _mm256_add_pd(sum2, xvec2);
        sum3 = _mm256_add_pd(sum3, xvec3);

        // increment
        xptr += DTYPE_UNROLLING_SIZE;
    }
    // combine sum0, sum1, sum2 and sum3
    __m256d partial_sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1),
                                        _mm256_add_pd(sum2, sum3));

    // handle teh last 4 - 15 elements
    const std::size_t remainder = end - xptr;
    if (remainder >= DTYPE_ELEMS_PER_REGISTER) {
        const std::size_t num_blocks = remainder / DTYPE_ELEMS_PER_REGISTER;
        for (std::size_t i = 0; i < num_blocks; ++i) {
            const __m256d xvec = _mm256_loadu_pd(xptr);
            sums = _mm256_add_pd(sums, xvec);
            xptr += DTYPE_ELEMS_PER_REGISTER;
        }
    }

    // perform a horizontal addition
    // [a,b,c,d] -> [c,d,a,b]
    // [a+c,b+d,c+a,d+d]
    // [(a+c)+(b+d),*,*,*]
    const __m256d perm = _mm256_permute2f128_pd(partial_sum, partial_sum, 0x01);
    const __m256d combine_sum = _mm256_add_pd(partial_sum, perm);
    const __m256d scalar_sum = _mm256_hadd_pd(combine_sum, combine_sum);
    total += _mm256_cvtsd_f64(scalar_sum);
    // sums = _mm256_add_pd(sums, _mm256_permute2f128_pd(sums, sums, 0x01));
    // sums = _mm256_hadd_pd(sums, sums);
    // total += _mm256_cvtsd_f64(sums);

    // process remaining elements
    const std::size_t tails = end - xptr;
    switch (tails) {
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
#endif // MATH_KERNELS_OPS_AVX_OPS_AVX_DOUBLE_HPP_
