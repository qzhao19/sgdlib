#ifndef MATH_KERNELS_OPS_SSE_OPS_SSE_FLOAT_HPP_
#define MATH_KERNELS_OPS_SSE_OPS_SSE_FLOAT_HPP_

#include "common/constants.hpp"
#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

#if defined(USE_SSE)
/**
 * @brief Sets all elements of a float array to a specified value using SSE instructions.
 */
inline void vecset_sse_float(float* x, const float c, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return ;

    // handle small size n < 4
    if (n < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = c;
        }
        return ;
    }

    // define a ptr points to x, end of bound
    float* xptr = x;
    const float* end = x + n;

    // handle aligned case
    // load scalar c into register and define aligned bound
    const __m128 scalar = _mm_set1_ps(c);
    const float* aligned_end = xptr + ((end - xptr) & ~3ULL);
    for (; xptr < aligned_end; xptr += 4) {
        _mm_storeu_ps(xptr, scalar);
    }

    // handle remaining elements
    const size_t remainder = end - xptr;
    switch (remainder) {
        case 3: xptr[2] = c; [[fallthrough]];
        case 2: xptr[1] = c; [[fallthrough]];
        case 1: xptr[0] = c;
        default: break;
    }
};

/**
 * @brief Copies an array of single-precision floating-point numbers using SSE4.2 instructions.
 */
void veccpy_sse_float(const float* x, std::size_t n, float* out) noexcept {
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
    float* outptr = out;
    const float* xptr = x;
    const float* end = x + n;

    // define aligned bound
    const float* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // main loop to process simd
    for (; xptr < aligned_end; xptr += 4, outptr += 4) {
        __m128 xvec = _mm_loadu_ps(xptr);
        _mm_storeu_ps(outptr, xvec);
    }

    // handle remaining elements
    const std::size_t remainder = end - xptr;
    switch (remainder){
        case 3: outptr[2] = xptr[2]; [[fallthrough]];
        case 2: outptr[1] = xptr[1]; [[fallthrough]];
        case 1: outptr[0] = xptr[0];
        default: break;
    }
};

/**
 * @brief Negates and copies elements of a float array using SSE intrinsics.
 */
void vecncpy_sse_float(const float* x, std::size_t n, float* out) noexcept {
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
    float* outptr = out;
    const float* xptr = x;
    const float* end = x + n;

    // define aligned bound and 32 mask
    const float* aligned_end = xptr + ((end - xptr) & ~3ULL);
    const __m128 sign_flip = _mm_castsi128_ps(
        _mm_set1_epi32(0x80000000)
    );

    // main loop to process simd
    for (; xptr < aligned_end; xptr += 4, outptr += 4) {
        __m128 xvec = _mm_loadu_ps(xptr);
        xvec = _mm_xor_ps(xvec, sign_flip);
        _mm_storeu_ps(outptr, xvec);
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
 * @brief Clips (clamps) elements of a float vector to specified [min, max] range using SSE4.2 intrinsics.
 *        Performs in-place modification: x[i] = min(max(x[i], min), max)
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

    // define ptr points to x and end of x
    float* xptr = x;
    const float* end = x + n;

    // load const values min/max to SIMD register
    // compute aligned bound of array
    const __m128 xmin = _mm_set1_ps(min);
    const __m128 xmax = _mm_set1_ps(max);
    const float* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // process the array in chunks of 4 elements
    for (; xptr < aligned_end; xptr += 4) {
        __m128 xvec = _mm_loadu_ps(xptr);
        xvec = _mm_min_ps(_mm_max_ps(xvec, xmin), xmax);
        _mm_storeu_ps(xptr, xvec);
    }

    const std::size_t remainder = end - xptr;
    switch (remainder) {
        case 3: xptr[2] = std::clamp(xptr[2], min, max); [[fallthrough]];
        case 2: xptr[1] = std::clamp(xptr[1], min, max); [[fallthrough]];
        case 1: xptr[0] = std::clamp(xptr[0], min, max);
        default: break;
    }
};

/**
 * Checks if a float vector contains any infinity values using SSE intrinsics.
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

    // define ptr to x
    const float* xptr = x;
    const float* end = x + n;

    // load const val to register
    const __m128 pos_inf = _mm_set1_ps(INF);
    const __m128 neg_inf = _mm_set1_ps(-INF);

    // compute aligned bound
    const float* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // processed the array in chunks of 4 elems
    for (; xptr < aligned_end; xptr += 4) {
        __m128 xvec = _mm_loadu_ps(xptr);
        __m128 cmp = _mm_or_ps(_mm_cmpeq_ps(xvec, pos_inf),
                               _mm_cmpeq_ps(xvec, neg_inf));

        if (_mm_movemask_ps(cmp) != 0) {
            return true;
        }
    }

    // process the rest of elems
    const std::size_t remainder = end - xptr;
    for (std::size_t i = 0; i < remainder; ++i) {
        if (std::isinf(xptr[i])) {
            return true;
        }
    }
    return false;
};

/**
 * @brief Computes the L2 norm (Euclidean norm) or squared L2 norm
 *        of a float vector using SSE intrinsics.
 */
inline float vecnorm2_sse_float(const float* x, std::size_t n, bool squared) noexcept {
    if (n == 0 || x == nullptr) return 0.0f;
    if (n < 4) {
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
    const float* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // main loop: memory aligned
    __m128 sum = _mm_setzero_ps();
    for (; xptr < aligned_end; xptr += 4) {
        __m128 xvec = _mm_loadu_ps(xptr);
        sum = _mm_add_ps(sum, _mm_mul_ps(xvec, xvec));
    }

    const __m128 shuffle = _mm_movehdup_ps(sum);  // [a,b,c,d] -> [b,b,d,d]
    const __m128 combine_sum = _mm_add_ps(sum, shuffle); // [a+b, b+b, c+d, d+d]
    const __m128 scalar_sum = _mm_add_ss(combine_sum, _mm_movehl_ps(combine_sum, combine_sum)); // add [a+b, c+d] -> [a+b+c+d]
    total += _mm_cvtss_f32(scalar_sum);

    // handle remaining elements
    const std::size_t remainder = end - xptr;
    switch (remainder) {
        case 3: total += xptr[2] * xptr[2]; [[fallthrough]];
        case 2: total += xptr[1] * xptr[1]; [[fallthrough]];
        case 1: total += xptr[0] * xptr[0];
        default: break;
    }

    return squared ? total : std::sqrt(total);
}

/**
 * @brief Computes the L1 norm (Manhattan norm) of a float array using SSE intrinsics.
 */
inline float vecnorm1_sse_float(const float* x, std::size_t n) noexcept {
    if (n == 0 || x == nullptr) return 0.0f;
    // handle small size n < 4
    if (n < 4) {
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
    const float* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // _mm_set1_epi32: set all elements to 0x7FFFFFFF,
    // contains 4 32bit int value, each value is 0x7FFFFFFF
    // _mm_castsi128_ps: convert int to float
    const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));

    __m128 sum = _mm_setzero_ps();
    for (; xptr < aligned_end; xptr += 4) {
        __m128 xvec = _mm_loadu_ps(xptr);
        // _mm_and_ps: bitwise '&' operation between mask and vec
        sum = _mm_add_ps(sum, _mm_and_ps(xvec, abs_mask));
    }

    // sum = [a,b,c,d] -> [b,b,d,d] -> [a+b, b+c, c+d, d+0] -> [a+b+c+d, c+d, d+0, 0]
    const __m128 shuffle = _mm_movehdup_ps(sum);  // [a,b,c,d] -> [b,b,d,d]
    const __m128 combine_sum = _mm_add_ps(sum, shuffle); // [a+b, b+b, c+d, d+d]
    // add [a+b, c+d] -> [a+b+c+d]
    const __m128 scalar_sum = _mm_add_ss(combine_sum, _mm_movehl_ps(combine_sum, combine_sum));
    total += _mm_cvtss_f32(scalar_sum);

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
 * @brief Scales a float array in-place by constant factor using SSE vectorization.
 *
 * Performs element-wise multiplication: x[i] = x[i] * c for all i in [0,n-1]
 *
 */
inline void vecscale_sse_float(const float* x,
                               const float c,
                               std::size_t n,
                               float* out) noexcept {
    // conditionn check
    if (x == nullptr || out == nullptr) return;
    if (n == 0 || c == 1.0f) return ;

    // for small size array
    if (n < 4) {
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
    const float* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // load constant c into register
    const __m128 scalar = _mm_set1_ps(c);

    // main SIMD loop
    for (; xptr < aligned_end; xptr += 4, outptr += 4) {
        __m128 xvec = _mm_loadu_ps(xptr);
        _mm_storeu_ps(outptr, _mm_mul_ps(xvec, scalar));
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
 * @brief Performs SIMD-accelerated vector scaling (element-wise multiplication) for
 *        single-precision floating-point arrays.
 *
 * Computes out[i] = x[i] * c for each element in the range [xbegin, xend) using SSE4.2 intrinsics.
 * Optimized for 32-bit floating-point data with automatic fallback to scalar operations.
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
 * This function is optimized for float arrays and handles aligned memory.
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
    float* outptr = out;
    const float* xptr = x;
    const float* yptr = y;
    const float* end = x + n;

    // compute aligned bound
    const float* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // main SIMD loop
    for (; xptr < aligned_end; xptr += 4, yptr += 4, outptr += 4) {
        __m128 xvec = _mm_loadu_ps(xptr);
        __m128 yvec = _mm_loadu_ps(yptr);
        _mm_storeu_ps(outptr, _mm_add_ps(xvec, yvec));
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
 * @brief Performs SIMD-accelerated vector addition with scaling for
 *        single-precision floating-point arrays. out[i] = x[i] * c + y[i]
 *        for each element using SSE4.2 intrinsics.
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
    float* outptr = out;
    const float* xptr = x;
    const float* yptr = y;
    const float* end = x + n;

    // define aligned bound of input x
    const float* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // load constant c into register
    const __m128 scalar = _mm_set1_ps(c);

    // start SIMD loop
    for (; xptr < aligned_end; xptr += 4, yptr += 4, outptr += 4) {
        __m128 xvec = _mm_loadu_ps(xptr);
        __m128 yvec = _mm_loadu_ps(yptr);
        _mm_storeu_ps(outptr, _mm_add_ps(_mm_mul_ps(xvec, scalar), yvec));
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
 * @brief Performs SIMD-accelerated element-wise subtraction of two single-precision
 *        floating-point arrays. Computes out[i] = x[i] - y[i] for each
 *        element using SSE4.2 instructions.
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
    float* outptr = out;
    const float* xptr = x;
    const float* yptr = y;
    const float* end = x + n;

    // define aligned bound of input x
    const float* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // start SIMD loop
    for (; xptr < aligned_end; xptr += 4, yptr += 4, outptr += 4) {
        __m128 xvec = _mm_loadu_ps(xptr);
        __m128 yvec = _mm_loadu_ps(yptr);
        _mm_storeu_ps(outptr, _mm_sub_ps(xvec, yvec));
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
 * @brief Computes the dot product of two single-precision floating-point vectors
 *        using SSE4.2 (Streaming SIMD Extensions) intrinsics.
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

    // define xptr and yptr point to x and y
    const float* xptr = x;
    const float* yptr = y;
    const float* end = x + n;

    // define aligned bound of input x
    const float* aligned_end = xptr + ((end - xptr) & ~3ULL);

    // load sum of vec to register
    __m128 sum = _mm_setzero_ps();

    // loop array x and y in chunks of 4 elems
    for (; xptr < aligned_end; xptr += 4, yptr += 4) {
        __m128 xvec = _mm_loadu_ps(xptr);
        __m128 yvec = _mm_loadu_ps(yptr);
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
    const __m128 shuffle = _mm_movehl_ps(sum, sum);
    const __m128 combine_sum = _mm_add_ps(sum, shuffle);
    const __m128 scalar_sum = _mm_add_ps(combine_sum, _mm_shuffle_ps(combine_sum, combine_sum, 0x55));
    float total = _mm_cvtss_f32(scalar_sum);

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
inline void vecmul_sse_float(const float* x,
                             const float* y,
                             std::size_t n,
                             std::size_t m,
                             float* out) noexcept {
    if (x == nullptr || y == nullptr) return ;
    if (out == nullptr) return ;
    if (n == 0 || m == 0) return ;
    if (m != n) return ;

    // handle small size case n < 4
    if (n < 4 && m < 4) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = x[i] * y[i];
        }
        return ;
    }

    // define xptr and yptr point to x and y
    float* outptr = out;
    const float* xptr = x;
    const float* yptr = y;
    const float* end = x + n;

    // define aligned bound of input x
    const float* aligned_end = xptr + ((end - xptr) & ~3ULL);

    for (; xptr < aligned_end; xptr += 4, yptr += 4, outptr += 4) {
        __m128 xvec = _mm_loadu_ps(xptr);
        __m128 yvec = _mm_loadu_ps(yptr);
        _mm_storeu_ps(outptr, _mm_mul_ps(xvec, yvec));
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
 * @brief Computes the accumulated sum of single-precision floating-point elements
 *        using SSE4.2 intrinsics
 */
inline float vecaccmul_sse_float(const float* xbegin,
                                 const float* xend,
                                 std::size_t n) noexcept {
    if (xbegin == nullptr || xend == nullptr) return 0.0f;
    if (xend <= xbegin) return 0.0f;
    const std::size_t m = static_cast<std::size_t>(xend - xbegin);
    if (n != m) return 0.0f;
    if (n == 0) return 0.0f;

    if (n < 4) {
        float acc = 0.0f;
        for (std::size_t i = 0; i < m; ++i) {
            acc += xbegin[i];
        }
        return acc;
    }

    const float* xptr = xbegin;
    const float* end = xbegin + n;
    const float* aligned_end = xptr + ((end - xptr) & ~3ULL);
    float total = 0.0f;

    __m128 sum = _mm_setzero_ps();
    for (; xptr < aligned_end; xptr += 4) {
        __m128 xvec = _mm_loadu_ps(xptr);
        sum = _mm_add_ps(sum, xvec);
    }

    const __m128 shuffle = _mm_movehdup_ps(sum);  // [a,b,c,d] -> [b,b,d,d]
    const __m128 combine_sum = _mm_add_ps(sum, shuffle); // [a+b, b+b, c+d, d+d]
    const __m128 scalar_sum = _mm_add_ss(combine_sum, _mm_movehl_ps(combine_sum, combine_sum)); // add [a+b, c+d] -> [a+b+c+d]
    total += _mm_cvtss_f32(scalar_sum);

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
#endif // MATH_KERNELS_OPS_SSE_OPS_SSE_FLOAT_HPP_
