#ifndef MATH_KERNELS_OPS_AVX_OPS_AVX_DOUBLE_HPP_
#define MATH_KERNELS_OPS_AVX_OPS_AVX_DOUBLE_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

/**
 *
 */
inline void vecset_avx_double(double* x, const double c, std::size_t n) noexcept {
    if (x == nullptr || n == 0) return;

    // handle small size n < 8
    if (n < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = c;
        }
        return;
    }

    // define a ptr points to x, end of bound
    double* xptr = x;
    const double* end = x + n;

    // load c into register, define aligned bound
    const __m256d scalar = _mm256_set1_pd(c);
    const double* aligned_end = xptr + ((end - xptr) & ~3ULL);

    for (; xptr < aligned_end; xptr += 4) {
        _mm256_store_pd(xptr, scalar);
    }

    // handle remaining elements
    const std::size_t remains = end - xptr;
    switch (remains) {
        case 3: xptr[2] = c; [[fallthrough]];
        case 2: xptr[1] = c; [[fallthrough]];
        case 1: xptr[0] = c;
        default: break;
    }
}

/**
 *
 */
void veccpy_avx_double(const double* x, double* out, std::size_t n) noexcept {
    if (n == 0) return ;
    if (x == nullptr) return ;
    if (out == nullptr) return ;
    // handle small size n < 4
    if (n < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i];
        }
        return ;
    }

    // define xptr, outptr and a ptr to end of x
    double* outptr = out;
    const double* xptr = x;
    const double* end = x + n;

    // define aligned bound
    const double* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // main loop to process simd
    for (; xptr < aligned_end; xptr += 4, outptr += 4) {
        // const __m128d xvec = _mm_load_pd(xptr);
        // _mm_store_pd(outptr, xvec);
        _mm256_store_pd(outptr, _mm256_load_pd(xptr));
    }

    // handle remaining elements
    const std::size_t remains = end - xptr;
    switch (remains) {
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
void vecncpy_avx_double(const double* x, double* out, std::size_t n) noexcept {
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

    // define aligned bound
    const double* aligned_end = xptr + ((end - xptr) & ~3ULL);
    const __m256d sign_flip = _mm256_set1_pd(-0.0);

    // main loop to process simd
    for (; xptr < aligned_end; xptr += 4, outptr += 4) {
        const __m256d xvec = _mm256_load_pd(xptr);
        const __m256d result = _mm256_xor_pd(xvec, sign_flip);
        _mm256_store_pd(outptr, result);
    }

    // handle remaining elements
    const std::size_t remains = end - xptr;
    switch (remains) {
        case 3: outptr[2] = -xptr[2]; [[fallthrough]];
        case 2: outptr[1] = -xptr[1]; [[fallthrough]];
        case 1: outptr[0] = -xptr[0];
        default: break;
    }
};

/**
 *
 */
inline void vecclip_avx_double(double* x, double min, double max, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return ;
    if (min > max) return ;

    // check if x is aligned to 16 bytes == 2 elems
    if (n < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            // x[i] = std::clamp(x[i], min, max);
            x[i] = (x[i] < min) ? min : (x[i] > max) ? max : x[i];
        }
        return ;
    }

    // Create an SSE register with all elements set to the min/max value
    const __m256d xmin = _mm256_set1_pd(min);
    const __m256d xmax = _mm256_set1_pd(max);

    // define ptr points to x and end of x
    double* xptr = x;
    const double* end = x + n;

    // compute aligned bound
    const double* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // Process the array in chunks of 2 elements using SSE intrinsics
    for (; xptr < aligned_end; xptr += 4) {
        __m256d vec = _mm256_load_pd(xptr);
        vec = _mm256_min_pd(_mm256_max_pd(vec, xmin), xmax);
        _mm256_store_pd(xptr, vec);
    }

    // handle remaining elements
    const std::size_t remains = end - xptr;
    switch (remains) {
        case 3: xptr[2] = (xptr[2] < min) ? min : (xptr[2] > max) ? max : xptr[2]; [[fallthrough]];
        case 2: xptr[1] = (xptr[1] < min) ? min : (xptr[1] > max) ? max : xptr[1]; [[fallthrough]];
        case 1: xptr[0] = (xptr[0] < min) ? min : (xptr[0] > max) ? max : xptr[0];
        default: break;
    }
};

/**
 *
 */
inline bool hasinf_avx_double(const double* x, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return false;

    // check if x has 2 elems
    if (n < 4) {
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

    // load x positif inf + x negative inf to SIMD register
    const __m256d pos_inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
    const __m256d neg_inf = _mm256_set1_pd(-std::numeric_limits<double>::infinity());

    // compute aligned bound
    const double* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // loop the array in chunks of 4 elements
    for (; xptr < aligned_end; xptr += 4) {
        const __m256d vec = _mm256_load_pd(xptr);
        const __m256d cmp = _mm256_or_pd(_mm256_cmp_pd(vec, pos_inf, _CMP_EQ_OQ),
                                         _mm256_cmp_pd(vec, neg_inf, _CMP_EQ_OQ));
        if (_mm256_movemask_pd(cmp) != 0) {
            return true;
        }
    }

    // process remaining elements
    if (end > xptr) {
        const std::size_t remains = end - xptr;
        for (std::size_t i = 0; i < remains; ++i) {
            if (std::isinf(xptr[i])) {
                return true;
            }
        }
    }
    return false;
};

/**
 *
 */
inline double vecnorm2_avx_double(const double* x,
                                  std::size_t n,
                                  bool squared) noexcept {
    if (n == 0 || x == nullptr) return 0.0;
    // For small arrays with less than or equal to two elements
    if (n < 4) {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            sum += x[i] * x[i];
        }
        return squared ? sum : std::sqrt(sum);
    }

    // compute aligned boundary
    const double* xptr = x;
    const double* end = x + n;
    double total = 0.0;

    // handle aligned case
    const double* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // init sum to 0
    __m256d sum = _mm256_setzero_pd();
    for (; xptr < aligned_end; xptr += 4) {
        // load 4 elements from x to SIMD register
        const __m256d vec = _mm256_load_pd(xptr);
        // compute sum = sum + vec * vec
        sum = _mm256_add_pd(sum, _mm256_mul_pd(vec, vec));
    }

    // perform a horizontal addition
    // [a,b,c,d] -> [c,d,a,b]
    // [a+c,b+d,c+a,d+d]
    // [(a+c)+(b+d),*,*,*]
    const __m256d perm = _mm256_permute2f128_pd(sum, sum, 0x01);
    const __m256d sums = _mm256_add_pd(sum, perm);
    const __m256d sumh = _mm256_hadd_pd(sums, sums);
    total += _mm256_cvtsd_f64(sumh);

    // process remaining elements
    const std::size_t remains = end - xptr;
    switch (remains) {
        case 3: total += xptr[2] * xptr[2]; [[fallthrough]];
        case 2: total += xptr[1] * xptr[1]; [[fallthrough]];
        case 1: total += xptr[0] * xptr[0];
        default: break;
    }
    return squared ? total : std::sqrt(total);
};

/**
 *
 */
inline double vecnorm1_avx_double(const double* x,
                                  std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return 0.0;
    // For small arrays with less than or equal to two elements
    if (n < 4) {
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

    // compute aligned bound
    const double* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // mask for the absolute value of the a double
    const __m256d abs_mask = _mm256_castsi256_pd(
        _mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));

    // init sum to 0
    __m256d sum = _mm256_setzero_pd();
    for (; xptr < aligned_end; xptr += 4) {
        __m256d vec = _mm256_load_pd(xptr);
        sum = _mm256_add_pd(sum, _mm256_and_pd(vec, abs_mask));
    }

    // perform a horizontal addition of the 4
    // channels' values in the AVX register
    const __m256d perm = _mm256_permute2f128_pd(sum, sum, 0x01);
    const __m256d sums = _mm256_add_pd(sum, perm);
    const __m256d sumh = _mm256_hadd_pd(sums, sums);
    total += _mm256_cvtsd_f64(sumh);

    // handle remaining elements
    const std::size_t remains = end - xptr;
    switch (remains) {
        case 3: total += std::abs(xptr[2]); [[fallthrough]];
        case 2: total += std::abs(xptr[1]); [[fallthrough]];
        case 1: total += std::abs(xptr[0]);
        default: break;
    }
    return total;
};

/**
 *
 */
inline void vecscale_avx_double(const double* x,
                                const double c,
                                std::size_t n,
                                double* out) noexcept {
    if (x == nullptr || out == nullptr) return ;
    if (n == 0 || c == 1.0) return ;

    // for small size n < 4
    if (n < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i] * c;
        }
        return ;
    }

     // define ptr points to x and end of x
     double* outptr = out;
     const double* xptr = x;
     const double* end = x + n;

    // define ptr points to x and aligned end
    const double* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // load constant into register
    const __m256d scalar = _mm256_set1_pd(c);

    // main SIMD loop
    for (; xptr < aligned_end; xptr += 4, outptr += 4) {
        const __m256d xvec = _mm256_load_pd(xptr);
        // const __m128d outx = _mm_mul_pd(xvec, scalar);
        _mm256_store_pd(outptr, _mm256_mul_pd(xvec, scalar));
    }

    // handle remaining elements
    const std::size_t remains = end - xptr;
    switch (remains) {
        case 3: outptr[2] = xptr[2] * c; [[fallthrough]];
        case 2: outptr[1] = xptr[1] * c; [[fallthrough]];
        case 1: outptr[0] = xptr[0] * c;
        default: break;
    }
};

/**
 *
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

    if (n < 4) {
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

    // define aligned bound
    const double* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // main SIMD loop
    for (; xptr < aligned_end; xptr += 4, yptr += 4, outptr += 4) {
        const __m256d xvec = _mm256_load_pd(xptr);
        const __m256d yvec = _mm256_load_pd(yptr);
        _mm256_store_pd(outptr, _mm256_add_pd(xvec, yvec));
    }

    // handle remaining elements
    const std::size_t remains = end - xptr;
    switch (remains) {
        case 3: outptr[2] = xptr[2] + yptr[2]; [[fallthrough]];
        case 2: outptr[1] = xptr[1] + yptr[1]; [[fallthrough]];
        case 1: outptr[0] = xptr[0] + yptr[0];
        default: break;
    }
};

/**
 *
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

    if (n < 4) {
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

    // define aligned bound
    const double* aligned_end = xptr + ((end - xptr) & ~1ULL);

    // load constant c into register
    const __m256d scalar = _mm256_set1_pd(c);

    for (; xptr < aligned_end; xptr += 4, yptr += 4, outptr += 4) {
        const __m256d xvec = _mm256_load_pd(xptr);
        const __m256d yvec = _mm256_load_pd(yptr);
        _mm256_store_pd(outptr, _mm256_add_pd(_mm256_mul_pd(xvec, scalar), yvec));
    }

    // handle remaining elements
    const std::size_t remains = end - xptr;
    switch (remains) {
        case 3: outptr[2] = c * xptr[2] + yptr[2]; [[fallthrough]];
        case 2: outptr[1] = c * xptr[1] + yptr[1]; [[fallthrough]];
        case 1: outptr[0] = c * xptr[0] + yptr[0];
        default: break;
    }
};

/**
 *
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
    const double* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // main SIMD loop
    for (; xptr < aligned_end; xptr += 4, yptr += 4, outptr += 4) {
        const __m256d xvec = _mm256_load_pd(xptr);
        const __m256d yvec = _mm256_load_pd(yptr);
        _mm256_store_pd(outptr, _mm256_sub_pd(xvec, yvec));
    }

    // handle remaining elements
    const std::size_t remains = end - xptr;
    switch (remains) {
        case 3: outptr[2] = xptr[2] - yptr[2]; [[fallthrough]];
        case 2: outptr[1] = xptr[1] - yptr[1]; [[fallthrough]];
        case 1: outptr[0] = xptr[0] - yptr[0];
        default: break;
    }
};







}
}
#endif // MATH_KERNELS_OPS_AVX_OPS_AVX_DOUBLE_HPP_
