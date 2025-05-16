#ifndef MATH_KERNELS_OPS_AVX_OPS_AVX_FLOAT_HPP_
#define MATH_KERNELS_OPS_AVX_OPS_AVX_FLOAT_HPP_

#include "common/constants.hpp"
#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

/**
 * @brief Sets all elements of a float array to a constant value using AVX instructions.
 */
inline void vecset_avx_float(float* x, const float c, std::size_t n) noexcept {
    if (x == nullptr || n == 0) return;

    // handle small size n < 16
    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = c;
        }
        return;
    }

    // define a ptr points to x, end of bound and aligned bound
    float* xptr = x;
    const float* end = x + n;

    // load scalar into register
    const __m256 scalar = _mm256_set1_ps(c);
    // handle aligned elements, process 16 elemts for each loop
    for (; xptr + UNROLLING_SIZE <= end; xptr += UNROLLING_SIZE) {
        _mm256_storeu_ps(xptr, scalar);
        _mm256_storeu_ps(xptr + 8, scalar);
    }

    // handle teh last 8 * 15 elements
    std::size_t remainder = end - xptr;
    if (remainder >= FTYPE_ELEMS_PER_BLOCK) {
        _mm256_storeu_ps(xptr, scalar);
        xptr += FTYPE_ELEMS_PER_BLOCK;
    }

    // handle the last 1 -7 remaining elemts
    if (end > xptr) {
        const std::size_t tails = end - xptr;
        for (std::size_t i = 0; i < tails; ++i) {
            xptr[i] = c;
        }
    }
};

/**
 * @brief Copies an array of float-precision floating-point numbers using AVX2 instructions.
 */
void veccpy_avx_float(const float* x, std::size_t n, float* out) noexcept {
    if (n == 0) return ;
    if (x == nullptr) return ;
    if (out == nullptr) return ;
    // handle small size n < 16
    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i];
        }
        return ;
    }

    // define xptr, outptr and a ptr to end of x
    float* outptr = out;
    const float* xptr = x;
    const float* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / UNROLLING_SIZE;

    // main loop to process simd
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        // load 16 xptr to 2 register
        __m256 xvec0 = _mm256_loadu_ps(xptr);
        __m256 xvec1 = _mm256_loadu_ps(xptr + 8);
        _mm256_storeu_ps(outptr, xvec0);
        _mm256_storeu_ps(outptr + 8, xvec1);
        // increment
        xptr += UNROLLING_SIZE;
        outptr += UNROLLING_SIZE;
    }

    // handle the last 8 - 15 elements
    std::size_t remainder = end - xptr;
    if (remainder >= FTYPE_ELEMS_PER_BLOCK) {
        const __m256 xvec = _mm256_loadu_ps(xptr);
        _mm256_storeu_ps(outptr, xvec);
        xptr += FTYPE_ELEMS_PER_BLOCK;
        outptr += FTYPE_ELEMS_PER_BLOCK;
    }

    // handle the last 1 -7 remaining elemts
    if (end > xptr) {
        const std::size_t tails = end - xptr;
        for (std::size_t i = 0; i < tails; ++i) {
            outptr[i] = xptr[i];
        }
    }
};

/**
 * @brief Negates and copies elements of a float array using AVX2 intrinsics.
 */
void vecncpy_avx_float(const float* x, std::size_t n, float* out) noexcept {
    if (n == 0) return ;
    if (x == nullptr) return ;
    if (out == nullptr) return ;
    // handle small size n < 16
    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = -x[i];
        }
        return ;
    }

    // define xptr, outptr and a ptr to end of x
    float* outptr = out;
    const float* xptr = x;
    const float* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / UNROLLING_SIZE;

    // define sign_flip mask
    const __m256 sign_flip = _mm256_castsi256_ps(
        _mm256_set1_epi32(0x80000000)
    );

    // main loop to process simd
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256 xvec0 = _mm256_loadu_ps(xptr);
        __m256 xvec1 = _mm256_loadu_ps(xptr + 8);

        xvec0 = _mm256_xor_ps(xvec0, sign_flip);
        xvec1 = _mm256_xor_ps(xvec1, sign_flip);

        _mm256_storeu_ps(outptr, xvec0);
        _mm256_storeu_ps(outptr + 8, xvec1);
        // increment
        xptr += UNROLLING_SIZE;
        outptr += UNROLLING_SIZE;
    }

    // handle teh last 8 - 15 elements
    std::size_t remainder = end - xptr;
    if (remainder >= FTYPE_ELEMS_PER_BLOCK) {
        __m256 xvec = _mm256_loadu_ps(xptr);
        xvec = _mm256_xor_ps(xvec, sign_flip);
        _mm256_storeu_ps(outptr, xvec);
        xptr += FTYPE_ELEMS_PER_BLOCK;
        outptr += FTYPE_ELEMS_PER_BLOCK;
    }

    // handle the last 1 -7 remaining elemts
    if (end > xptr) {
        const std::size_t tails = end - xptr;
        for (std::size_t i = 0; i < tails; ++i) {
            outptr[i] = -xptr[i];
        }
    }
};

/**
 * @brief Clips elements of a float array to specified range using AVX intrinsics
 */
inline void vecclip_avx_float(float* x, float min, float max, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return ;
    if (min > max) return ;
    // handle small size n < 16
    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = std::clamp(x[i], min, max);
        }
        return;
    }

    // define ptr points to x and end of x
    float* xptr = x;
    const float* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / UNROLLING_SIZE;

    // load const values min/max to SIMD register
    const __m256 xmin = _mm256_set1_ps(min);
    const __m256 xmax = _mm256_set1_ps(max);

    // process the array in chunks of 16 elements
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256 xvec0 = _mm256_loadu_ps(xptr);
        __m256 xvec1 = _mm256_loadu_ps(xptr + 8);
        xvec0 = _mm256_min_ps(_mm256_max_ps(xvec0, xmin), xmax);
        xvec1 = _mm256_min_ps(_mm256_max_ps(xvec1, xmin), xmax);
        _mm256_storeu_ps(xptr, xvec0);
        _mm256_storeu_ps(xptr + 8, xvec1);

         // increment
        xptr += UNROLLING_SIZE;
    }

    // handle teh last 8 - 15 elements
    std::size_t remainder = end - xptr;
    if (remainder >= FTYPE_ELEMS_PER_BLOCK) {
        __m256 xvec = _mm256_loadu_ps(xptr);
        xvec = _mm256_min_ps(_mm256_max_ps(xvec, xmin), xmax);
        _mm256_storeu_ps(xptr, xvec);
        xptr += FTYPE_ELEMS_PER_BLOCK;
    }

    // handle the last 1 - 7 remaining elemts
    if (end > xptr) {
        const std::size_t tails = end - xptr;
        for (std::size_t i = 0; i < tails; ++i) {
             xptr[i] = std::clamp(xptr[i], min, max);
        }
    }
};

/**
 * @brief Checks for infinite values in a float array using AVX intrinsics
 */
inline bool hasinf_avx_float(const float* x, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return false;

    // handle small size n < 16
    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isinf(x[i])) {
                return true;
            }
        }
        return false;
    }

    // define ptr to x
    const float* xptr = x;
    const float* end = x + n;
    // loop unrolling
    const std::size_t num_unrolls = n / UNROLLING_SIZE;

    // load const val to register
    const __m256 pos_inf = _mm256_set1_ps(INF);
    const __m256 neg_inf = _mm256_set1_ps(-INF);
    // processed the array in chunks of 8 elems
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256 xvec0 = _mm256_loadu_ps(xptr);
        __m256 xvec1 = _mm256_loadu_ps(xptr + 8);
        __m256 cmp0 = _mm256_or_ps(_mm256_cmp_ps(xvec0, pos_inf, _CMP_EQ_OQ),
                                   _mm256_cmp_ps(xvec0, neg_inf, _CMP_EQ_OQ));
        __m256 cmp1 = _mm256_or_ps(_mm256_cmp_ps(xvec1, pos_inf, _CMP_EQ_OQ),
                                   _mm256_cmp_ps(xvec1, neg_inf, _CMP_EQ_OQ));
        if (_mm256_movemask_ps(_mm256_or_ps(cmp0, cmp1)) != 0) {
            return true;
        }

        xptr += UNROLLING_SIZE;
    }

    // handle teh last 8 - 15 elements
    std::size_t remainder = end - xptr;
    if (remainder >= FTYPE_ELEMS_PER_BLOCK) {
        const __m256 xvec = _mm256_loadu_ps(xptr);
        const __m256 cmp = _mm256_or_ps(_mm256_cmp_ps(xvec, pos_inf, _CMP_EQ_OQ),
                                        _mm256_cmp_ps(xvec, neg_inf, _CMP_EQ_OQ));
        if (_mm256_movemask_ps(cmp) != 0) {
            return true;
        }
        xptr += FTYPE_ELEMS_PER_BLOCK;
    }

    // handle the last 1 - 7 remaining elemts
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
 * @brief Computes the squared L2 norm (Euclidean norm) of a single-precision vector using AVX2 intrinsics.
 */
inline float vecnorm2_avx_float(const float* x, std::size_t n, bool squared) noexcept {
    if (n == 0 || x == nullptr) return 0.0f;
    if (n < 16) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            sum += x[i] * x[i];
        }
        return squared ? sum : std::sqrt(sum);
    }

    // define ptr points to x and end of x
    const float* xptr = x;
    const float* end = x + n;
    float total = 0.0f;

    // loop unrolling
    const std::size_t num_unrolls = n / UNROLLING_SIZE;

    // main loop: memory unaligned
    // init sum0, sum1 to 0
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256 xvec0 = _mm256_loadu_ps(xptr);
        __m256 xvec1 = _mm256_loadu_ps(xptr + 8);
        // sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(xvec0, xvec0));
        // sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(xvec1, xvec1));
        // sum += x * x
        sum0 = _mm256_fmadd_ps(xvec0, xvec0, sum0);
        sum1 = _mm256_fmadd_ps(xvec1, xvec1, sum1);
        // increment
        xptr += UNROLLING_SIZE;
    }

    // combine sum0 and sum1
    __m256 sums = _mm256_add_ps(sum0, sum1);

    // handle teh last 8 - 15 elements
    std::size_t remainder = end - xptr;
    if (remainder >= FTYPE_ELEMS_PER_BLOCK) {
        __m256 xvec = _mm256_loadu_ps(xptr);
        // sums = _mm256_add_ps(sums, _mm256_mul_ps(xvec, xvec));
        sums = _mm256_fmadd_ps(xvec, xvec, sums);
        xptr += FTYPE_ELEMS_PER_BLOCK;
    }

    // exact avx horizontal sum
    // [a,b,c,d,e,f,g,h] + [a,b,c,d,e,f,g,h]
    // [a,b,c,d,e,f,g,h] -> [a,b,c,d]
    // [a,b,c,d,e,f,g,h] -> [e,f,g,h]
    // [a+e,b+f,c+g,d+h]
    // [a+e,b+f,c+g,d+h] -> [b+f,b+f,d+h,d+h]
    // [(a+e)+(b+f),(b+f)+(b+f),(c+g)+(d+h),(d+h)+(d+h)]
    //   [(a+e)+(b+f),(b+f)+(b+f),(c+g)+(d+h),(d+h)+(d+h)]
    // + [(c+g)+(d+h),(d+h)+(d+h),(c+g)+(d+h),(d+h)+(d+h)]
    // =                   [(a+e)+(b+f)+(c+g)+(d+h),*,*,*]
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sums),
                               _mm256_extractf128_ps(sums, 1));
    sum128 = _mm_add_ps(sum128, _mm_movehdup_ps(sum128));
    sum128 = _mm_add_ss(sum128, _mm_movehl_ps(sum128, sum128));
    // extract first element
    total += _mm_cvtss_f32(sum128);

    // handle the last 1 - 7 remaining elemts
    if (end > xptr) {
        const std::size_t tails = end - xptr;
        for (std::size_t i = 0; i < tails; ++i) {
            total += xptr[i] * xptr[i];
        }
    }
    return squared ? total : std::sqrt(total);
};


/**
 * @brief Computes the squared L1 norm of a single-precision vector using AVX2 intrinsics.
 */
inline float vecnorm1_avx_float(const float* x, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return 0.0f;
    // handle small size n < 16
    if (n < 16) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            sum += std::abs(x[i]);
        }
        return sum;
    }

    // define ptr point to x and end of x
    const float* xptr = x;
    const float* end = x + n;
    float total = 0.0f;

    // loop unrolling
    const std::size_t num_unrolls = n / UNROLLING_SIZE;

    // _mm256_set1_epi32: generate a sequence which contains 8
    // 32bit int value, each value is 0x7FFFFFFF
    // _mm256_castsi256_ps: convert int to float
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256 xvec0 = _mm256_loadu_ps(xptr);
        __m256 xvec1 = _mm256_loadu_ps(xptr + 8);
        // _mm_and_ps: bitwise '&' operation between mask and vec
        sum0 = _mm256_add_ps(sum0, _mm256_and_ps(xvec0, abs_mask));
        sum1 = _mm256_add_ps(sum1, _mm256_and_ps(xvec1, abs_mask));
        xptr += UNROLLING_SIZE;
    }

    // combine sum0 and sum1
    __m256 sums = _mm256_add_ps(sum0, sum1);

    // handle teh last 8 - 15 elements
    std::size_t remainder = end - xptr;
    if (remainder >= FTYPE_ELEMS_PER_BLOCK) {
        __m256 xvec = _mm256_loadu_ps(xptr);
        sums = _mm256_add_ps(sums, _mm256_and_ps(xvec, abs_mask));
        xptr += FTYPE_ELEMS_PER_BLOCK;
    }

    // [a,b,c,d,e,f,g,h] -> [a,b,c,d]
    // [a,b,c,d,e,f,g,h] -> [e,f,g,h]
    // [a+e,b+f,c+g,d+h]
    // [a+e,b+f,c+g,d+h] -> [b+f,b+f,d+h,d+h]
    // [(a+e)+(b+f), (b+f)+(b+f), (c+g)+(d+h), (d+h)+(d+h)]
    //   [(a+e)+(b+f),(b+f)+(b+f),(c+g)+(d+h),(d+h)+(d+h)]
    // + [(c+g)+(d+h),(d+h)+(d+h),(c+g)+(d+h),(d+h)+(d+h)]
    // =                   [(a+e)+(b+f)+(c+g)+(d+h),*,*,*]
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sums),
                               _mm256_extractf128_ps(sums, 1));
    sum128 = _mm_add_ps(sum128, _mm_movehdup_ps(sum128));
    sum128 = _mm_add_ss(sum128, _mm_movehl_ps(sum128, sum128));
    total += _mm_cvtss_f32(sum128);

    // handle 1 - 7 last remaing elements
    if (end > xptr) {
        const std::size_t tails = end - xptr;
        for (std::size_t i = 0; i < tails; ++i) {
            total += std::abs(xptr[i]);
        }
    }
    return total;
};

/**
 * @brief Scales a float array in-place by constant factor using AVX vectorization.
 */
inline void vecscale_avx_float(const float* x,
                               const float c,
                               std::size_t n,
                               float* out) noexcept {
    // conditionn check
    if (x == nullptr || out == nullptr) return;
    if (n == 0 || c == 1.0f) return ;

    // for small size array
    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i] * c;
        }
        return;
    }

    // define ptr points to x and end of x
    // avoid modify origin output ptr out
    float* outptr = out;
    const float* xptr = x;
    const float* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / UNROLLING_SIZE;

    // load constant c into register
    const __m256 scalar = _mm256_set1_ps(c);
    // main SIMD loop
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256 xvec0 = _mm256_loadu_ps(xptr);
        __m256 xvec1 = _mm256_loadu_ps(xptr + 8);
        _mm256_storeu_ps(outptr, _mm256_mul_ps(xvec0, scalar));
        _mm256_storeu_ps(outptr + 8, _mm256_mul_ps(xvec1, scalar));
        xptr += UNROLLING_SIZE;
        outptr += UNROLLING_SIZE;
    }

    // handle teh last 8 - 15 elements
    std::size_t remainder = end - xptr;
    if (remainder >= FTYPE_ELEMS_PER_BLOCK) {
        __m256 xvec = _mm256_loadu_ps(xptr);
        _mm256_storeu_ps(outptr, _mm256_mul_ps(xvec, scalar));
        xptr += FTYPE_ELEMS_PER_BLOCK;
        outptr += FTYPE_ELEMS_PER_BLOCK;
    }

    // handle last 1 - 7 remaining elements
    if (end > xptr) {
        const std::size_t tails = end - xptr;
        for (std::size_t i = 0; i < tails; ++i) {
            outptr[i] = xptr[i] * c;
        }
    }
};

/**
 *
 */
inline void vecscale_avx_float(const float* xbegin,
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
    vecscale_avx_float(xbegin, c, n, out);
};

/**
 *
 */
inline void vecadd_avx_float(const float* x,
                             const float* y,
                             const std::size_t n,
                             const std::size_t m,
                             float* out) noexcept {
    if (x == nullptr || y == nullptr) return ;
    if (out == nullptr) return ;
    if (n == 0 || m == 0) return ;
    if (m != n) return ;

    // handle small size array n < 16
    if (n < 16) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i] + y[i];
        }
        return ;
    }

    // define ptr points to x
    float* outptr = out;
    const float* xptr = x;
    const float* yptr = y;
    const float* end = x + n;

    // loop unrolling
    const std::size_t num_unrolls = n / UNROLLING_SIZE;

    // main SIMD loop
    for (std::size_t i = 0; i < num_unrolls; ++i) {
        __m256 xvec0 = _mm256_loadu_ps(xptr);
        __m256 yvec0 = _mm256_loadu_ps(yptr);
        __m256 xvec1 = _mm256_loadu_ps(xptr + 8);
        __m256 yvec1 = _mm256_loadu_ps(yptr + 8);
        _mm256_storeu_ps(outptr, _mm256_add_ps(xvec0, yvec0));
        _mm256_storeu_ps(outptr + 8, _mm256_add_ps(xvec1, yvec1));
        xptr += UNROLLING_SIZE;
        yptr += UNROLLING_SIZE;
        outptr += UNROLLING_SIZE;
    }

    // handle teh last 8 - 15 elements
    std::size_t remainder = end - xptr;
    if (remainder >= FTYPE_ELEMS_PER_BLOCK) {
        __m256 xvec = _mm256_loadu_ps(xptr);
        __m256 yvec = _mm256_loadu_ps(yptr);
        _mm256_storeu_ps(outptr, _mm256_add_ps(xvec, yvec));
        xptr += FTYPE_ELEMS_PER_BLOCK;
        yptr += FTYPE_ELEMS_PER_BLOCK;
        outptr += FTYPE_ELEMS_PER_BLOCK;
    }

    // handle 1 - 7 remaining elements
    if (end > xptr) {
        const std::size_t remains = end - xptr;
        for (std::size_t i = 0; i < remains; ++i) {
            outptr[i] = xptr[i] + yptr[i];
        }
    }
};

/**
 *
 */
inline void vecadd_avx_float(const float* x,
                             const float* y,
                             const float c,
                             const std::size_t n,
                             const std::size_t m,
                             float* out) noexcept {
    if (x == nullptr || y == nullptr) return ;
    if (out == nullptr) return ;
    if (n == 0 || m == 0) return ;
    if (m != n) return ;

    if (n < 8) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = c * x[i] + y[i];
        }
        return ;
    }

    // define xptr and yptr point to x and y
    float* outptr = out;
    const float* xptr = x;
    const float* yptr = y;
    const float* end = x + n;

    // define aligned bound of input x
    const float* aligned_end = xptr + ((end - xptr) & ~7ULL);

    // load constant c into register
    const __m256 scalar = _mm256_set1_ps(c);

    // start SIMD loop
    for (; xptr < aligned_end; xptr += 8, yptr += 8, outptr += 8) {
        const __m256 xvec = _mm256_loadu_ps(xptr);
        const __m256 yvec = _mm256_loadu_ps(yptr);
        _mm256_storeu_ps(outptr, _mm256_add_ps(_mm256_mul_ps(xvec, scalar), yvec));
    }

    // handle remaining elements
    if (end > xptr) {
        const std::size_t remains = end - xptr;
        for (std::size_t i = 0; i < remains; ++i) {
            outptr[i] = c * xptr[i] + yptr[i];
        }
    }
};

/**
 *
 */
inline void vecdiff_avx_float(const float* x,
                              const float* y,
                              const std::size_t n,
                              const std::size_t m,
                              float* out) noexcept {
    if (x == nullptr || y == nullptr) return ;
    if (out == nullptr) return ;
    if (n == 0 || m == 0) return ;
    if (m != n) return ;

    // handle small size n < 4
    if (n < 8) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i] - y[i];
        }
        return ;
    }

    // define xptr and yptr point to x and y
    float* outptr = out;
    const float* xptr = x;
    const float* yptr = y;
    const float* end = x + n;

    // define aligned bound of input x
    const float* aligned_end = xptr + ((end - xptr) & ~7ULL);

    // start SIMD loop
    for (; xptr < aligned_end; xptr += 8, yptr += 8, outptr += 8) {
        const __m256 xvec = _mm256_loadu_ps(xptr);
        const __m256 yvec = _mm256_loadu_ps(yptr);
        _mm256_storeu_ps(outptr, _mm256_sub_ps(xvec, yvec));
    }

    // handle remaining elements
    if (end > xptr) {
        const std::size_t remains = end - xptr;
        for (std::size_t i = 0; i < remains; ++i) {
            outptr[i] = xptr[i] - yptr[i];
        }
    }
};

/**
 *
 */
inline float vecdot_avx_float(const float* x,
                              const float* y,
                              std::size_t n,
                              std::size_t m) noexcept {

    if (x == nullptr || y == nullptr) return 0.0f;
    if (n != m) return 0.0f;
    if (n == 0) return 0.0f;

    // handle small size case n < 4
    if (n < 8 && m < 8) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    // define xptr and yptr point to x and y
    const float* xptr = x;
    const float* yptr = y;
    const float* end = x + n;

    // define aligned bound of input x
    const float* aligned_end = xptr + ((end - xptr) & ~7ULL);

    // load sum of vec to register
    __m256 sum = _mm256_setzero_ps();

    // loop array x and y in chunks of 8 elems
    for (; xptr < aligned_end; xptr += 8, yptr += 8) {
        const __m256 xvec = _mm256_loadu_ps(xptr);
        const __m256 yvec = _mm256_loadu_ps(yptr);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(xvec, yvec));
    }

    //
    const __m128 vlow = _mm256_castps256_ps128(sum);     // [a,b,c,d,e,f,g,h] -> [a,b,c,d]
    const __m128 vhigh = _mm256_extractf128_ps(sum, 1);  // [a,b,c,d,e,f,g,h] -> [e,f,g,h]
    const __m128 sum128 = _mm_add_ps(vlow, vhigh);       // [a+e,b+f,c+g,d+h]
    const __m128 shuf = _mm_movehdup_ps(sum128);         // [a+e,b+f,c+g,d+h] -> [b+f,b+f,d+h,d+h]
    const __m128 sumh = _mm_add_ps(sum128, shuf);        // [(a+e)+(b+f), (b+f)+(b+f), (c+g)+(d+h), (d+h)+(d+h)]
    //   [(a+e)+(b+f),(b+f)+(b+f),(c+g)+(d+h),(d+h)+(d+h)]
    // + [(c+g)+(d+h),(d+h)+(d+h),(c+g)+(d+h),(d+h)+(d+h)]
    // =                   [(a+e)+(b+f)+(c+g)+(d+h),*,*,*]
    const __m128 sums = _mm_add_ss(sumh, _mm_movehl_ps(sumh, sumh));
    float total = _mm_cvtss_f32(sums);

    // handle remaining elements
    if (end > xptr) {
        const std::size_t remains = end - xptr;
        for (std::size_t i = 0; i < remains; ++i) {
            total += xptr[i] * yptr[i];
        }
    }
    return total;
};



}
}
#endif // MATH_KERNELS_OPS_AVX_OPS_AVX_FLOAT_HPP_
