#ifndef MATH_MATH_OPS_HPP_
#define MATH_MATH_OPS_HPP_

#include "common/logging.hpp"
#include "common/prereqs.hpp"
#include "math/kernels/ops_ansi.hpp"
#include "math/kernels/ops_avx.hpp"
#include "math/kernels/ops_sse.hpp"


namespace sgdlib {
namespace detail {

/**
 * @brief Sets all elements of a vector to a constant value with hardware acceleration
 *
 * @tparam T Element type (float/double)
 *
 * @param[in,out] x Target vector to be modified
 * @param[in] c Constant value to fill
 *
 * @note Implementation selection logic:
 * - Uses AVX vectorization when USE_AVX is defined
 * - Falls back to SSE vectorization when USE_SSE is defined
 * - Defaults to ANSI implementation otherwise
 * @note Marked as inline for performance-critical operations
 */
template<typename T>
inline void vecset(std::vector<T>& x, const T c) {
    static_assert(std::is_arithmetic_v<T>,
        "vecset requires arithmetic types (e.g. int, float, double)");

    if (x.empty()) {
        THROW_INVALID_ERROR("vecset: input vector cannot be empty");
    }

    std::size_t n = x.size();

#if defined(USE_SSE)
    vecset_sse(x.data(), c, n);
#elif defined(USE_AVX)
    vecset_avx(x.data(), c, n);
#else
    vecset_ansi(x, c);
#endif
};

/**
 * @brief Copies vector contents
 *
 * @tparam T Element type (float/double)
 *
 * @param[in] x Source vector to copy from
 * @param[out] out Target vector to receive copy
 *
 * @note Implementation selection:
 * - Only use to ANSI memcpy
 *
 * @note Requires out vector to be pre-allocated with sufficient size
 */
template<typename T>
inline void veccpy(const std::vector<T>& x, std::vector<T>& out) {
    static_assert(std::is_arithmetic_v<T>,
        "veccpy requires arithmetic types (e.g. int, float, double)");

    if (x.empty() || out.empty()) {
        THROW_INVALID_ERROR("veccpy: input/output vector cannot be empty");
    }

    if (x.size() != out.size()) {
        THROW_INVALID_ERROR("veccpy: requires x.size() == out.size()");
    }

    veccpy_ansi(x, out);
};

/**
 * @brief Copies and negates vector elements with hardware acceleration
 *
 * @tparam T Element type (float/double)
 * @param[in] x Source vector to copy from
 * @param[out] out Target vector storing negated values
 *
 * @note Implementation selection:
 * - Uses AVX when USE_AVX defined
 * - Falls back to SSE when USE_SSE defined
 * - Defaults to ANSI std::transform otherwise
 *
 * @throws Requires out.size() >= x.size()
 */
template<typename T>
inline void vecncpy(const std::vector<T>& x, std::vector<T>& out) {
    static_assert(std::is_arithmetic_v<T>,
        "vecncpy requires arithmetic types (e.g. int, float, double)");

    if (x.empty() || out.empty()) {
        THROW_INVALID_ERROR("vecncpy: input/output vector cannot be empty");
    }

    if (x.size() > out.size()) {
        THROW_INVALID_ERROR("vecncpy: requires out.size() >= x.size()");
    }
    std::size_t n = x.size();

#if defined(USE_SSE)
    vecncpy_sse(x.data(), n, out);
#elif defined(USE_AVX)
    vecncpy_avx(x.data(), n, out);
#else
    vecncpy_ansi(x, out);
#endif
};

/**
 * @brief Clips vector elements to [min,max] range with hardware acceleration
 *
 * @tparam T Arithmetic type (int/float/double)
 *
 * @param[in,out] x Target vector to be clipped
 * @param[in] min Lower bound of clipping range
 * @param[in] max Upper bound of clipping range
 *
 * @note Implementation priority:
 * 1. AVX vectorization when USE_AVX defined
 * 2. SSE vectorization when USE_SSE defined
 * 3. ANSI std::transform otherwise
 *
 * @throws:
 * - Empty input vector
 * - min > max value range
 */
template<typename T>
inline void vecclip(std::vector<T>& x, T min, T max) {
    static_assert(
        std::is_arithmetic_v<T>,
        "vecclip requires arithmetic types (e.g. int, float, double)"
    );

    if (x.empty()) {
        THROW_INVALID_ERROR("vecclip: input vector cannot be empty");
    }

    if (min > max) {
        THROW_INVALID_ERROR(
            "vecclip: min (" + std::to_string(min) +
            ") must be <= max (" + std::to_string(max) + ")"
        );
    }
    std::size_t n = x.size();

#if defined(USE_SSE)
    vecclip_sse(x.data(), min, max, n);
#elif defined(USE_AVX)
    vecclip_avx(x.data(), min, max, n);
#else
    vecclip_ansi(x, min, max);
#endif
};

/**
 * @brief Checks for infinite values in a vector using hardware acceleration
 *
 * @tparam T Arithmetic type (float/double)
 *
 * @param[in] x Input vector to be checked
 * @return bool True if any element is ±infinity, false otherwise
 *
 * @note Implementation priority:
 * 1. AVX vectorization when USE_AVX defined
 * 2. SSE vectorization when USE_SSE defined
 * 3. ANSI std::any_of otherwise
 *
 * @throws std::invalid_argument For empty input vector
 */
template<typename T>
inline bool hasinf(const std::vector<T>& x) {
    static_assert(std::is_floating_point_v<T>,
        "hasinf requires floating-point types (e.g. float, double)"
    );

    if (x.empty()) {
        THROW_INVALID_ERROR("hasinf: input vector cannot be empty");
    }

    std::size_t n = x.size();
    bool has_inf = false;
#if defined(USE_SSE)
    has_inf = hasinf_sse(x.data(), n);
#elif defined(USE_AVX)
    has_inf = hasinf_avx(x.data(), n);
#else
    has_inf = hasinf_ansi(x);
#endif
    return has_inf;
};

/**
 * @brief Computes L2 norm of a vector with hardware acceleration
 *
 * @tparam T Floating-point type (float/double)
 *
 * @param[in] x Input vector to compute norm from
 * @param[in] squared When true returns squared norm (skip sqrt)
 * @return T Computed norm value
 *
 * @note Implementation selection:
 * - AVX vectorization when USE_AVX defined
 * - SSE vectorization when USE_SSE defined
 * - ANSI std::inner_product otherwise
 *
 * @throws std::invalid_argument For empty input vector
 */
template<typename T>
inline T vecnorm2(const std::vector<T>& x, bool squared = false) {
    static_assert(std::is_floating_point_v<T>,
        "vecnorm2 requires floating-point types (e.g. float, double)"
    );

    if (x.empty()) {
        THROW_INVALID_ERROR("vecnorm2: input vector cannot be empty");
    }

    T norm2;
    std::size_t n = x.size();
#if defined(USE_SSE)
    norm2 = vecnorm2_sse(x.data(), n, squared);
#elif defined(USE_AVX)
    norm2 = vecnorm2_avx(x.data(), n, squared);
#else
    norm2 = vecnorm2_ansi(x, squared);
#endif
    return norm2;
}

/**
 * @brief Computes L1 norm of a vector with hardware acceleration
 *
 * @tparam T Floating-point type (float/double)
 *
 * @param[in] x Input vector to compute norm from
 * @param[in] squared When true returns squared norm (skip sqrt)
 * @return T Computed norm value
 *
 * @note Implementation selection:
 * - AVX vectorization when USE_AVX defined
 * - SSE vectorization when USE_SSE defined
 * - ANSI std::inner_product otherwise
 *
 * @throws std::invalid_argument For empty input vector
 */
template<typename T>
inline T vecnorm1(const std::vector<T>& x) {
    static_assert(std::is_floating_point_v<T>,
        "vecnorm1 requires floating-point types (e.g. float, double)"
    );

    if (x.empty()) {
        THROW_INVALID_ERROR("vecnorm1: input vector cannot be empty");
    }

    T norm1;
    std::size_t n = x.size();
#if defined(USE_SSE)
    norm1 = vecnorm1_sse(x.data(), n);
#elif defined(USE_AVX)
    norm1 = vecnorm1_avx(x.data(), n);
#else
    norm1 = vecnorm1_ansi(x);
#endif
    return norm1;
}

/**
 * @brief Scales vector elements by a constant with hardware acceleration
 *
 * @tparam T Floating-point type (float/double)
 *
 * @param[in] x Source vector to be scaled
 * @param[in] c Scaling factor
 * @param[out] out Target vector storing scaled values
 *
 * @note Implementation priority:
 * 1. AVX vectorization when USE_AVX defined
 * 2. SSE vectorization when USE_SSE defined
 * 3. ANSI std::transform otherwise
 *
 * @throws std::invalid_argument For:
 * - Empty input/output vectors
 * - Size mismatch between input and output vectors
 */
template<typename T>
inline void vecscale(const std::vector<T>& x,
                     const T& c,
                     std::vector<T>& out) {
    static_assert(std::is_floating_point_v<T>,
        "vecscale requires floating-point types (e.g. float, double)");

    if (x.empty() || out.empty()) {
        THROW_INVALID_ERROR("vecscale: input/output vector cannot be empty");
    }

    if (x.size() != out.size()) {
        THROW_INVALID_ERROR("vecscale: requires out.size() == x.size()");
    }
    std::size_t n = x.size();

#if defined(USE_SSE)
    vecscale_sse(x.data(), c, n, out.data());
#elif defined(USE_AVX)
    vecscale_avx(x.data(), c, n, out.data());
#else
    vecscale_ansi(x, c, out);
#endif
}

/**
 * @brief Scales vector elements in a range [xbegin, xend) by a constant with hardware acceleration
 *
 * @tparam T Floating-point type (float/double)
 *
 * @param[in] xbegin Pointer to the first element in the range
 * @param[in] xend Pointer past the last element in the range
 * @param[in] c Scaling factor
 * @param[out] out Target vector storing scaled values
 *
 * @note Implementation priority:
 * 1. AVX vectorization when USE_AVX defined
 * 2. SSE vectorization when USE_SSE defined
 * 3. ANSI std::transform otherwise
 *
 * @throws std::invalid_argument For:
 * - Null pointer input
 * - Invalid range (xbegin >= xend)
 * - Empty input range
 * - Size mismatch between input range and output vector
 */
template<typename T>
inline void vecscale(const T* xbegin,
                     const T* xend,
                     const T& c,
                     std::vector<T>& out) noexcept {
    static_assert(std::is_floating_point_v<T>,
        "vecscale requires floating-point types (e.g. float, double)");

    if (xbegin == nullptr || xend == nullptr) {
        THROW_INVALID_ERROR("vecscale: input pointers cannot be null (xbegin="
                            + std::to_string(reinterpret_cast<uintptr_t>(xbegin))
                            + ", xend="
                            + std::to_string(reinterpret_cast<uintptr_t>(xend)) + ")");
    }

    if (xbegin >= xend) {
        THROW_INVALID_ERROR("vecscale: invalid range [xbegin, xend) with size="
                            + std::to_string(xend - xbegin));
    }
    const std::size_t n = static_cast<std::size_t>(xend - xbegin);
    if (n == 0) {
        THROW_INVALID_ERROR("vecscale: empty input range");
    }
    if (n != out.size()) {
        THROW_INVALID_ERROR("vecscale: requires out.size() == len(xend - xbegin)");
    }

#if defined(USE_SSE)
    vecscale_sse(xbegin, xend, c, n, out.data());
#elif defined(USE_AVX)
    vecscale_avx(xbegin, xend, c, n, out.data());
#else
    vecscale_ansi(xbegin, xend, c, out);
#endif

};

/**
 * @brief Performs vector addition with hardware acceleration
 *
 * @tparam T Floating-point type (float/double)
 *
 * @param[in] x First input vector
 * @param[in] y Second input vector
 * @param[out] out Output vector storing element-wise sum x + y
 *
 * @note Implementation priority:
 * 1. AVX vectorization when USE_AVX defined
 * 2. SSE vectorization when USE_SSE defined
 * 3. ANSI std::transform otherwise
 *
 * @throws std::invalid_argument For:
 * - Empty input vectors
 * - Size mismatch between input vectors
 * - Output vector size mismatch
 */
template<typename T>
inline void vecadd(const std::vector<T>& x,
                  const std::vector<T>& y,
                  std::vector<T>& out) {

    static_assert(std::is_floating_point_v<T>,
        "vecadd requires floating-point types (e.g. float, double)");

    if (x.empty() || y.empty()) {
        THROW_INVALID_ERROR("vecadd: input x, y vector cannot be empty");
    }

    if (out.empty()) {
        THROW_INVALID_ERROR("vecadd: output vector cannot be empty");
    }

    std::size_t n = x.size();
    std::size_t m = y.size();
    if (m != n || n != out.size()) {
        THROW_INVALID_ERROR("vecadd: requires x.size() == y.size() == out.size()");
    }

#if defined(USE_SSE)
    vecadd_sse(x.data(), y.data(), n, m, out.data());
#elif defined(USE_AVX)
    vecadd_avx(x.data(), y.data(), n, m, out.data());
#else
    vecadd_ansi(x, y, out);
#endif
}

/**
 * @brief Performs scaled vector addition with hardware acceleration
 *
 * @tparam T Floating-point type (float/double)
 *
 * @param[in] x First input vector
 * @param[in] y Second input vector
 * @param[in] c Scaling factor applied to x
 * @param[out] out Output vector storing element-wise result x*c + y
 *
 * @note Implementation priority:
 * 1. AVX vectorization when USE_AVX defined
 * 2. SSE vectorization when USE_SSE defined
 * 3. ANSI std::transform otherwise
 *
 * @throws std::invalid_argument For:
 * - Empty input vectors
 * - Size mismatch between input vectors
 * - Output vector size mismatch
 */
template<typename T>
inline void vecadd(const std::vector<T>& x,
                  const std::vector<T>& y,
                  const T& c,
                  std::vector<T>& out) {

    static_assert(std::is_floating_point_v<T>,
        "vecadd requires floating-point types (e.g. float, double)");

    if (x.empty() || y.empty()) {
        THROW_INVALID_ERROR("vecadd: input x, y vector cannot be empty");
    }

    if (out.empty()) {
        THROW_INVALID_ERROR("vecadd: output vector cannot be empty");
    }

    std::size_t n = x.size();
    std::size_t m = y.size();
    if (m != n || n != out.size()) {
        THROW_INVALID_ERROR("vecadd: requires x.size() == y.size() == out.size()");
    }

#if defined(USE_SSE)
    vecadd_sse(x.data(), y.data(), c, n, m, out.data());
#elif defined(USE_AVX)
    vecadd_avx(x.data(), y.data(), c, n, m, out.data());
#else
    vecadd_ansi(x, y, c, out);
#endif
}

/**
 * @brief Performs vector subtraction with hardware acceleration
 *
 * @tparam T Floating-point type (float/double)
 *
 * @param[in] x First input vector
 * @param[in] y Second input vector
 * @param[out] out Output vector storing element-wise difference x - y
 *
 * @note Implementation priority:
 * 1. AVX vectorization when USE_AVX defined
 * 2. SSE vectorization when USE_SSE defined
 * 3. ANSI std::transform otherwise
 *
 * @throws std::invalid_argument For:
 * - Empty input vectors
 * - Size mismatch between input vectors
 * - Output vector size mismatch
 */
template<typename T>
inline void vecdiff(const std::vector<T>& x,
                    const std::vector<T>& y,
                    std::vector<T>& out) {
    static_assert(std::is_floating_point_v<T>,
        "vecdiff requires floating-point types (e.g. float, double)");

    if (x.empty() || y.empty()) {
        THROW_INVALID_ERROR("vecdiff: input x, y vector cannot be empty");
    }

    if (out.empty()) {
        THROW_INVALID_ERROR("vecdiff: output vector cannot be empty");
    }

    std::size_t n = x.size();
    std::size_t m = y.size();
    if (m != n || n != out.size()) {
        THROW_INVALID_ERROR("vecdiff: requires x.size() == y.size()");
    }
    if (n != out.size()) {
        THROW_INVALID_ERROR("vecdiff: requires x.size() == out.size()");
    }
#if defined(USE_SSE)
    vecdiff_sse(x.data(), y.data(), n, m, out.data());
#elif defined(USE_AVX)
    vecdiff_avx(x.data(), y.data(), n, m, out.data());
#else
    vecdiff_ansi(x, y, out);
#endif

}

/**
 * @brief Computes dot product of two vectors with hardware acceleration
 *
 * @tparam T Floating-point type (float/double)
 * @param[in] x First input vector
 * @param[in] y Second input vector
 * @return T Computed dot product value
 *
 * @note Implementation priority:
 * 1. AVX vectorization when USE_AVX defined
 * 2. SSE vectorization when USE_SSE defined
 * 3. ANSI std::inner_product otherwise
 *
 * @throws std::invalid_argument For:
 * - Empty input vectors
 * - Size mismatch between input vectors
 */
template<typename T>
inline T vecdot(const std::vector<T>& x,
                const std::vector<T>& y) {
    static_assert(std::is_floating_point_v<T>,
        "vecdot requires floating-point types (e.g. float, double)");

    if (x.empty() || y.empty()) {
        THROW_INVALID_ERROR("vecdot: input x, y vector cannot be empty");
    }

    std::size_t n = x.size();
    std::size_t m = y.size();
    if (m != n) {
        THROW_INVALID_ERROR("vecdot: requires x.size() == y.size()");
    }

    T prod;
#if defined(USE_SSE)
    prod = vecdot_sse(x.data(), y.data(), n, m);
#elif defined(USE_AVX)
    prod = vecdot_avx(x.data(), y.data(), n, m);
#else
    prod = vecdot_ansi(x, y, out);
#endif
    return prod;
}

/**
 * @brief Performs element-wise vector multiplication with hardware acceleration
 *
 * @tparam T Floating-point type (float/double)
 * 
 * @param[in] x First input vector
 * @param[in] y Second input vector
 * @param[out] out Output vector storing element-wise product x * y
 *
 * @note Implementation priority:
 * 1. AVX vectorization when USE_AVX defined
 * 2. SSE vectorization when USE_SSE defined
 * 3. ANSI std::transform otherwise
 *
 * @throws std::invalid_argument For:
 * - Empty input vectors
 * - Size mismatch between input vectors
 * - Output vector size mismatch
 */
template<typename T>
inline void vecmul(const std::vector<T>& x,
                   const std::vector<T>& y,
                   std::vector<T>& out) {
    static_assert(std::is_floating_point_v<T>,
        "vecmul requires floating-point types (e.g. float, double)");

    if (x.empty() || y.empty()) {
        THROW_INVALID_ERROR("vecmul: input x, y vector cannot be empty");
    }

    std::size_t n = x.size();
    std::size_t m = y.size();
    if (m != n) {
        THROW_INVALID_ERROR("vecmul: requires x.size() == y.size()");
    }

    if (n != out.size()) {
        THROW_INVALID_ERROR("vecmul: requires x.size() == out.size()");
    }
#if defined(USE_SSE)
    vecmul_sse(x.data(), y.data(), n, m, out.data());
#elif defined(USE_AVX)
    vecmul_avx(x.data(), y.data(), n, m, out.data());
#else
    vecmul_ansi(x, y, out);
#endif
}


/**
 * @brief Clip (limit) the values in a vector.
 *
 * @tparam Type The type of elements in the vector.
 *
 * @param x vector containing elements to clip.
 * @param min, max minimum and maximum value
 *
 * @note The function is marked as inline, which is suitable for small functions
 *       to reduce the overhead of function calls.
*/
template<typename Type>
inline void clip(std::vector<Type>& x, Type min, Type max) {
    if (min > max) {
        THROW_INVALID_ERROR("a_min must be less than or equal to a_max.");
    }
    std::transform(std::begin(x), std::end(x), std::begin(x),
        [=] (auto i) {
            return std::clamp(i, min, max);
        }
    );
};

/**
 * @brief Clips the value to the specified range [min, max]
 *
 * @tparam Type The type of elements for inputs.
 *
 * @param x The input value to be clipped.
 * @param min, max minimum and maximum value
 *
 * @note The function is marked as inline, which is suitable for small functions
 *       to reduce the overhead of function calls.
*/
template<typename Type>
inline void clip(Type& x, Type min, Type max) {
    if (min > max) {
        THROW_INVALID_ERROR("a_min must be less than or equal to a_max.");
    }
    x = std::max(min, std::min(x, max));
};

/**
 * @brief check if any element of vector is infinite.
 *
 * @tparam Type The type of elements in the vector.
 *
 * @param x vector containing elements to check infinity.
 *
 * @note The function is marked as inline, which is suitable for small functions
 *       to reduce the overhead of function calls.
*/
template<typename Type>
inline bool isinf(const std::vector<Type>& x) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (std::isinf(x[i])) {
            return true;
        }
    }
    return false;
};

/**
 * @brief check if the given value represents infinity
 *
 * @tparam Type The type of elements for input.
 *
 * @param x value to check infinity.
 *
 * @note The function is marked as inline, which is suitable for small functions
 *       to reduce the overhead of function calls.
 */
template<typename Type>
inline bool isinf(const Type& x) {
    if (std::isinf(x)) {
        return true;
    }
    return false;
};

/**
 * @brief calculate the L2 norm of a vector.
 *
 * @tparam Type The type of elements in the vector.
 *
 * @param x a vector of type T.
 * @return The L2 norm of the vector as a custome type.
 *
 * @note The function is marked as inline, which is suitable for small functions
 *       to reduce the overhead of function calls.
 */
template<typename Type>
inline Type sqnorm2(const std::vector<Type>& x, bool squared) {
    Type norm2 = std::inner_product(x.begin(),
                                    x.end(),
                                    x.begin(),
                                    static_cast<Type>(0.0));
    if (squared) {
        norm2 = std::sqrt(norm2);
    }

    return norm2;
};

/**
 * @brief calculate the L1 norm (Manhattan distance) of a vector.
 *
 * @tparam Type The type of elements in the vector.
 *
 * @param x a vector of type T.
 * @return The L1 norm of the vector as a T.
 *
*/
template<typename Type>
inline Type norm1(const std::vector<Type>& x) {
    Type norm = 0;
    for (const Type& value : x) {
        norm += std::abs(value);
    }
    return norm;
};

/**
 * @brief Applies a scalar multiplication operation to a vector.
 *
 * @tparam Type The type of elements in the vector.
 *
 * @param[in,out] x vector of type T, which will be scaled by the scalar 'c'.
 * @param[in] scalar constant scalar value
 *
 * @note The function is marked as inline, which is suitable for small functions
 *       to reduce the overhead of function calls.
*/
template<typename Type>
inline void dot(std::vector<Type>& x, const Type c) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] *= c;
    }
};

/**
 * @brief Applies a scalar multiplication operation to a vector.
 * It computes the sum of the products of all elements within
 * the iterator range from begin to end with a constant c.
 *
 * @tparam Type The type of elements in the vector.
 * @tparam IterType The type of iterator, with a default of std::vector<Type>::const_iterator.
 *
 * @param begin The beginning iterator pointing to the first element
 *      of the vector to calculate the dot product.
 * @param end The ending iterator pointing to the first element following begin.
 * @param c The constant to be multiplied with the elements of the vector.
 * @param out The reference to a vector that stores the result of the dot product.
 *
 * @note The function is marked as inline, which is suitable for small functions
 *       to reduce the overhead of function calls.
 * @note The use of templates and default iterator types allows the function
 *       to be used with different types of vectors and iterators.
 *
*/
template<typename Type,
         typename IterType = typename std::vector<Type>::const_iterator>
inline void dot(IterType begin, IterType end,
                const Type c,
                std::vector<Type>& out) {
    if(std::distance(begin, end) != out.size()){
        THROW_INVALID_ERROR("Output vector size is insufficient.");
    }
    std::transform(begin, end,
                   out.begin(),
                   [c](const Type& elem) {
                        return elem * c;
                   });
};

/**
 * @brief Applies a scalar multiplication operation to a vector.
 * It computes the sum of the products of all elements within
 * the iterator range from begin to end with a constant c.
 *
 * @tparam Type The type of elements in the vector.
 *
 * @param[in] x vector of type T,
 * @param[in] y vector of type T,
 * @param out The reference to a vector that stores the result of the dot product.
 *
 * @note The function is marked as inline, which is suitable for small functions
 *       to reduce the overhead of function calls.
 *
 */
template <typename Type>
inline void dot(const std::vector<Type>& x,
                const std::vector<Type>& y,
                Type& out) {
    if (x.size() != y.size()) {
        THROW_INVALID_ERROR("Vectors must have the same size.");
    }
    out = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
}

template<typename Type>
void inline add(const std::vector<Type>& x,
                const std::vector<Type>& y,
                std::vector<Type>& out) {
    // check
    if (x.size() != y.size()) {
        THROW_INVALID_ERROR("Vectors must have the same size.");
    }

    out.resize(x.size());
    std::transform(x.begin(), x.end(), y.begin(), out.begin(), std::plus<Type>());
}

/**
 * @brief Computes the row-wise norms of a vector.
 *
 * This function calculates the norms of each row in a given vector `x`,
 * The result can be squared or not, based on the `squared` flag.
 *
 * @tparam Type A floating-point or arithmetic type that supports necessary operations.
 * @param x A constant reference to the input vector
 * @param squared A boolean flag that determines whether to compute the squared norms.
 *                - If true, the function computes the squared Euclidean norm:
 *                  \( \sum_{i=1}^{n} x_i^2 \)
 *                - If false, the function computes the Euclidean norm (non-squared), which
 *                  is the square root of the squared norm.
 * @param out A reference to the output vector where the computed norms will be stored.
 *
 * @note The function assumes that `x` represents a matrix with its size is ncols * nrows.
 *
*/
template<typename Type>
void row_norms(const std::vector<Type>& x,
               bool squared,
               std::vector<Type>& out) {

    std::size_t num_elems = x.size();
    std::size_t nrows = out.size();
    std::size_t ncols = num_elems / nrows;
    std::vector<Type> elem_prod(num_elems);

    // compute x * x
    std::transform(x.begin(), x.end(), elem_prod.begin(),
                    [](const Type& value) {
                        return value * value;
                    });

    // compute prefix sum of elem_prod
    std::vector<Type> prefix_sum(num_elems);
    std::partial_sum(elem_prod.begin(), elem_prod.end(), prefix_sum.begin());

    // compute the sum of every nth element
    std::size_t count = 0;
    for (std::size_t i = 0; i < num_elems; i += ncols) {
        std::size_t end = std::min(i + ncols, num_elems);
        if (count == 0) {
            out[count] = prefix_sum[end - 1];
        } else {
            out[count] = prefix_sum[end - 1] - prefix_sum[i - 1];
        }
        count += 1;
    }

    if (!squared) {
        std::transform(out.begin(), out.end(), out.begin(),
                    [](const Type& value) {
                        return std::sqrt(value);
                    });
    }
};


/**
 * @brief Computes the column-wise norms of a vector.
 *
 * This function calculates the norms of columns in a vector, treating it as a 2D matrix.
 * The norms can be either squared or not, based on the 'squared' parameter.
 *
 * @tparam Type The data type of the vector elements.
 * @param[in] x The input vector to compute norms from, treated as a 2D matrix.
 * @param[in] squared If true, computes squared norms; if false, computes regular norms.
 * @param[out] out The output vector to store the computed norms. Its size determines the number of columns.
 *
 * @note The function assumes that the size of 'x' is divisible by the size of 'out'.
 */
template<typename Type>
void col_norms(const std::vector<Type>& x,
               bool squared,
               std::vector<Type>& out) {

    std::size_t num_elems = x.size();
    std::size_t ncols = out.size();
    std::size_t nrows = num_elems / ncols;

    std::size_t j = 0;
    while (j < ncols) {
        Type sum_sq = 0.0;
        for (std::size_t i = 0; i < nrows; ++i) {
            Type val = x[j + i * ncols];
            sum_sq += val * val;
        }
        out[j] = sum_sq;
        ++j;
    }

    if (!squared) {
        std::transform(out.begin(), out.end(), out.begin(),
                    [](const Type& value) {
                        return std::sqrt(value);
                    });
    }
};


} // namespace detail
} // namespace sgdlib

#endif // MATH_MATH_OPS_HPP_
