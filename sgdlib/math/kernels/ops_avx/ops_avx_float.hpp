#ifndef MATH_KERNELS_OPS_AVX_OPS_AVX_FLOAT_HPP_
#define MATH_KERNELS_OPS_AVX_OPS_AVX_FLOAT_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

/**
 *
 */
inline void vecset_avx_float(float* x, const float c, std::size_t n) noexcept {
    if (x == nullptr || n == 0) return;

    // handle small size n < 8
    if (n < 8) {
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = c;
        }
        return;
    }

    // define a ptr points to x, end of bound and aligned bound
    float* xptr = x;
    const float* end = x + n;

    // define aligned bound
    const __m256 scalar = _mm256_set1_ps(c);
    const float* aligned_end = xptr + ((end - xptr) & ~7ULL);

    // handle aligned elements, process 8 elemts for eaach loop
    for (; xptr < aligned_end; xptr += 8) {
        _mm256_store_ps(xptr, scalar);
    }

    // handle remaining elemts
    if (end > xptr) {
        const std::size_t remains = end - xptr;
        for (std::size_t i = 0; i < remains; ++i) {
            xptr[i] = c;
        }
    }
};

/**
 *
 */
void veccpy_avx_float(const float* x, float* out, std::size_t n) noexcept {
    if (n == 0) return ;
    if (x == nullptr) return ;
    if (out == nullptr) return ;
    // handle small size n < 8
    if (n < 8) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i];
        }
        return ;
    }

    // define xptr, outptr and a ptr to end of x
    float* outptr = out;
    const float* xptr = x;
    const float* end = x + n;

    // define aligned bound
    const float* aligned_end = xptr + ((end - xptr) & ~7ULL);

    // main loop to process simd
    for (; xptr < aligned_end; xptr += 8, outptr += 8) {
        // const __m128 xvec = _mm_load_ps(xptr);
        // _mm_store_ps(outptr, xvec);
        _mm256_store_ps(outptr, _mm256_load_ps(xptr));
    }

    // handle remaining elemts
    if (end > xptr) {
        const std::size_t remains = end - xptr;
        for (std::size_t i = 0; i < remains; ++i) {
            outptr[i] = xptr[i];
        }
    }
};

/**
 *
 */
void vecncpy_avx_float(const float* x, float* out, std::size_t n) noexcept {
    if (n == 0) return ;
    if (x == nullptr) return ;
    if (out == nullptr) return ;
    // handle small size n < 8
    if (n < 8) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = -x[i];
        }
        return ;
    }

    // define xptr, outptr and a ptr to end of x
    float* outptr = out;
    const float* xptr = x;
    const float* end = x + n;

    // define aligned bound
    const float* aligned_end = xptr + ((end - xptr) & ~7ULL);
    const __m256 sign_flip = _mm256_set1_ps(-0.0f);

    // main loop to process simd
    for (; xptr < aligned_end; xptr += 8, outptr += 8) {
        const __m256 xvec = _mm256_load_ps(xptr);
        const __m256 result = _mm256_xor_ps(xvec, sign_flip);
        _mm256_store_ps(outptr, result);
    }

    // handle remaining elements
    if (end > xptr) {
        const std::size_t remains = end - xptr;
        for (std::size_t i = 0; i < remains; ++i) {
            outptr[i] = -xptr[i];
        }
    }
};

/**
 *
 */
inline void vecclip_avx_float(float* x, float min, float max, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return ;
    if (min > max) return ;
    // handle small size n < 8
    if (n < 8) {
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = std::clamp(x[i], min, max);
        }
        return;
    }

    // define ptr points to x and end of x
    float* xptr = x;
    const float* end = x + n;

    // load const values min/max to SIMD register
    // compute aligned bound of array
    const __m256 xmin = _mm256_set1_ps(min);
    const __m256 xmax = _mm256_set1_ps(max);
    const float* aligned_end = xptr + ((end - xptr) & ~7ULL);

    // process the array in chunks of 8 elements
    for (; xptr < aligned_end; xptr += 8) {
        __m256 vec = _mm256_load_ps(xptr);
        vec = _mm256_min_ps(_mm256_max_ps(vec, xmin), xmax);
        _mm256_store_ps(xptr, vec);
    }

    // handle remaining elements
    if (end > xptr) {
        const std::size_t remains = end - xptr;
        for (std::size_t i = 0; i < remains; ++i) {
            xptr[i] = std::clamp(xptr[i], min, max);
        }
    }
};

/**
 *
 */
inline bool hasinf_avx_float(const float* x, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return false;

    // handle small size n < 7
    if (n < 7) {
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

    // load const val to register
    const __m256 pos_inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    const __m256 neg_inf = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

    // compute aligned bound
    const float* aligned_end = xptr + ((end - xptr) & ~7ULL);

    // processed the array in chunks of 8 elems
    for (; xptr < aligned_end; xptr += 8) {
        const __m256 vec = _mm256_load_ps(xptr);
        const __m256 cmp = _mm256_or_ps(_mm256_cmp_ps(vec, pos_inf, _CMP_EQ_OQ),
                                        _mm256_cmp_ps(vec, neg_inf, _CMP_EQ_OQ));

        if (_mm256_movemask_ps(cmp) != 0) {
            return true;
        }
    }

    // handle remaining elements
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
inline float vecnorm2_avx_float(const float* x, std::size_t n, bool squared) noexcept {
    if (n == 0 || x == nullptr) return 0.0f;
    if (n < 8) {
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

    // compute aligned bound
    const float* aligned_end = xptr + ((end - xptr) & ~7ULL);

    // main loop: memory aligned
    // init sum to 0
    __m256 sum = _mm256_setzero_ps();
    for (; xptr < aligned_end; xptr += 8) {
        const __m256 vec = _mm256_load_ps(xptr);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(vec, vec));
    }

    // exact avx horizontal sum
    const __m128 vlow = _mm256_castps256_ps128(sum);     // [a,b,c,d,e,f,g,h] -> [a,b,c,d]
    const __m128 vhigh = _mm256_extractf128_ps(sum, 1);  // [a,b,c,d,e,f,g,h] -> [e,f,g,h]
    const __m128 sum128 = _mm_add_ps(vlow, vhigh);    // [a+e,b+f,c+g,d+h]
    const __m128 shuf = _mm_movehdup_ps(sum128);      // [a+e,b+f,c+g,d+h] -> [b+f,b+f,d+h,d+h]
    const __m128 sumh = _mm_add_ps(sum128, shuf);     // [(a+e)+(b+f),(b+f)+(b+f),(c+g)+(d+h),(d+h)+(d+h)]
    //   [(a+e)+(b+f),(b+f)+(b+f),(c+g)+(d+h),(d+h)+(d+h)]
    // + [(c+g)+(d+h),(d+h)+(d+h),(c+g)+(d+h),(d+h)+(d+h)]
    // =                   [(a+e)+(b+f)+(c+g)+(d+h),*,*,*]
    const __m128 sums = _mm_add_ss(sumh, _mm_movehl_ps(sumh, sumh));
    // extract first element
    total += _mm_cvtss_f32(sums);

    // handle remaining elements
    if (end > xptr) {
        const std::size_t remains = end - xptr;
        for (std::size_t i = 0; i < remains; ++i) {
            total += xptr[i] * xptr[i];
        }
    }
    return squared ? total : std::sqrt(total);
}


/**
 *
 */
inline float vecnorm1_avx_float(const float* x, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return 0.0f;
    // handle small size n < 4
    if (n < 8) {
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

    // compute aligned bound = xsize - xsize % 4
    const float* aligned_end = xptr + ((end - xptr) & ~7ULL);

    // _mm256_set1_epi32: generate a sequence which contains 8
    // 32bit int value, each value is 0x7FFFFFFF
    // _mm256_castsi256_ps: convert int to float
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    __m256 sum = _mm256_setzero_ps();
    for (; xptr < aligned_end; xptr += 8) {
        const __m256 vec = _mm256_load_ps(xptr);
        // _mm_and_ps: bitwise '&' operation between mask and vec
        sum = _mm256_add_ps(sum, _mm256_and_ps(vec, abs_mask));
    }

    const __m128 vlow = _mm256_castps256_ps128(sum);     // [a,b,c,d,e,f,g,h] -> [a,b,c,d]
    const __m128 vhigh = _mm256_extractf128_ps(sum, 1);  // [a,b,c,d,e,f,g,h] -> [e,f,g,h]
    const __m128 sum128 = _mm_add_ps(vlow, vhigh);       // [a+e,b+f,c+g,d+h]
    const __m128 shuf = _mm_movehdup_ps(sum128);         // [a+e,b+f,c+g,d+h] -> [b+f,b+f,d+h,d+h]
    const __m128 sumh = _mm_add_ps(sum128, shuf);           // [(a+e)+(b+f), (b+f)+(b+f), (c+g)+(d+h), (d+h)+(d+h)]
    //   [(a+e)+(b+f),(b+f)+(b+f),(c+g)+(d+h),(d+h)+(d+h)]
    // + [(c+g)+(d+h),(d+h)+(d+h),(c+g)+(d+h),(d+h)+(d+h)]
    // =                   [(a+e)+(b+f)+(c+g)+(d+h),*,*,*]
    const __m128 sums = _mm_add_ss(sumh, _mm_movehl_ps(sumh, sumh));
    total += _mm_cvtss_f32(sums);

    // handle remaing elements
    if (end > xptr) {
        const std::size_t remains = end - xptr;
        for (std::size_t i = 0; i < remains; ++i) {
            total += std::abs(xptr[i]);
        }
    }
    return total;
};

/**
 *
 */
inline void vecscale_avx_float(const float* x,
                               const float c,
                               std::size_t n,
                               float* out) noexcept {
    // conditionn check
    if (x == nullptr || out == nullptr) return;
    if (n == 0 || c == 1.0f) return ;

    // for small size array
    if (n < 8) {
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

    // compute aligned bound
    const float* aligned_end = xptr + ((end - xptr) & ~7ULL);

    // load constant c into register
    const __m256 scalar = _mm256_set1_ps(c);

    // main SIMD loop
    for (; xptr < aligned_end; xptr += 8, outptr += 8) {
        const __m256 xvec = _mm256_load_ps(xptr);
        _mm256_store_ps(outptr, _mm256_mul_ps(xvec, scalar));
    }

    // handle remaining elements
    if (end > xptr) {
        const std::size_t remains = end - xptr;
        for (std::size_t i = 0; i < remains; ++i) {
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

    // handle small size array n < 4
    if (n < 8) {
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

    // compute aligned bound
    const float* aligned_end = xptr + ((end - xptr) & ~7ULL);

    // main SIMD loop
    for (; xptr < aligned_end; xptr += 8, yptr += 8, outptr += 8) {
        const __m256 xvec = _mm256_load_ps(xptr);
        const __m256 yvec = _mm256_load_ps(yptr);
        _mm256_store_ps(outptr, _mm256_add_ps(xvec, yvec));
    }
    // handle remaining elements
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
        const __m256 xvec = _mm256_load_ps(xptr);
        const __m256 yvec = _mm256_load_ps(yptr);
        _mm256_store_ps(outptr, _mm256_add_ps(_mm256_mul_ps(xvec, scalar), yvec));
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
        const __m256 xvec = _mm256_load_ps(xptr);
        const __m256 yvec = _mm256_load_ps(yptr);
        _mm256_store_ps(outptr, _mm256_sub_ps(xvec, yvec));
    }

    // handle remaining elements
    if (end > xptr) {
        const std::size_t remains = end - xptr;
        for (std::size_t i = 0; i < remains; ++i) {
            outptr[i] = xptr[i] - yptr[i];
        }
    }
};






}
}
#endif // MATH_KERNELS_OPS_AVX_OPS_AVX_FLOAT_HPP_
