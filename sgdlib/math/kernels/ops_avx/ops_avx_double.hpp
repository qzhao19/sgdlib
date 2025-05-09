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






}
}
#endif // MATH_KERNELS_OPS_AVX_OPS_AVX_DOUBLE_HPP_
