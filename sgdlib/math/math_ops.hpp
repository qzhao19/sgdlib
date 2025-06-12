#ifndef MATH_MATH_OPS_HPP_
#define MATH_MATH_OPS_HPP_

#include "common/logging.hpp"
#include "common/prereqs.hpp"
#include "math/kernels/ops_ansi.hpp"
#include "math/kernels/ops_avx.hpp"
#include "math/kernels/ops_sse.hpp"
#include "data/continuous_dataset.hpp"

namespace sgdlib {
namespace detail {

/**
 * @brief Allocates memory for a vector with automatic lifetime management
 *
 * @tparam T Element type of the vector
 * @param n Number of elements to allocate
 *
 * @return std::unique_ptr<T[], void(*)(void*)>
 *         Smart pointer managing the allocated memory block with custom deleter
 *
 * @throws std::bad_alloc If memory allocation fails
 *
 * @note Key features:
 * - Provides exception-safe memory allocation
 * - Automatic memory deallocation via free()
 * - Zero-initializes allocated memory
 * - Memory alignment matches underlying SIMD implementation when enabled
 * - Custom deleter handles proper memory release
 *
 * @usage
 * auto buffer = vecalloc<float>(1024); // Allocates 1024 aligned floats
 */
template<typename T>
inline std::unique_ptr<T[], void(*)(void*)> vecalloc(std::size_t n) {
    void* ptr;
#if defined(USE_SSE) || defined(USE_AVX)
    std::size_t bytes = n * sizeof(T);
    if (bytes % MEMORY_ALIGNMENT != 0) {
        bytes += MEMORY_ALIGNMENT - (bytes % MEMORY_ALIGNMENT);
    }
    ptr = std::aligned_alloc(MEMORY_ALIGNMENT, bytes);
#else
    ptr = std::malloc(n * sizeof(T));
#endif
    if (!ptr) {
        throw std::bad_alloc();
    }
    std::memset(ptr, 0, n * sizeof(T));
    return std::unique_ptr<T[], void(*)(void*)>(
        static_cast<T*>(ptr),
        [](void* p) { free(p); }
    );
};

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
    static_assert(std::is_floating_point_v<T>,
        "vecset requires floating-point types (e.g. float, double)");

    if (x.empty()) {
        THROW_INVALID_ERROR("vecset: input vector cannot be empty");
    }

    std::size_t n = x.size();

#if defined(USE_SSE)
    vecset_sse<T>(x.data(), c, n);
#elif defined(USE_AVX)
    vecset_avx<T>(x.data(), c, n);
#else
    vecset_ansi<T>(x, c);
#endif
};

template<>
inline void vecset(std::vector<std::size_t>& x, const std::size_t c) {
    std::fill(x.begin(), x.end(), c);
}

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
    static_assert(std::is_floating_point_v<T>,
        "veccpy requires floating-point types (e.g. float, double)");

    if (x.empty() || out.empty()) {
        THROW_INVALID_ERROR("veccpy: input/output vector cannot be empty");
    }

    if (x.size() != out.size()) {
        THROW_INVALID_ERROR("veccpy: requires x.size() == out.size()");
    }

    veccpy_ansi<T>(x, out);
};

/**
 * @brief Copies and negates vector elements with hardware acceleration
 *        out[i] = -x[i]
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
    static_assert(std::is_floating_point_v<T>,
        "vecncpy requires floating-point types (e.g. float, double)");

    if (x.empty() || out.empty()) {
        THROW_INVALID_ERROR("vecncpy: input/output vector cannot be empty");
    }

    if (x.size() > out.size()) {
        THROW_INVALID_ERROR("vecncpy: requires out.size() >= x.size()");
    }
    std::size_t n = x.size();

#if defined(USE_SSE)
    vecncpy_sse<T>(x.data(), n, out.data());
#elif defined(USE_AVX)
    vecncpy_avx<T>(x.data(), n, out.data());
#else
    vecncpy_ansi<T>(x, out);
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
        std::is_floating_point_v<T>,
        "vecclip requires floating-point types (e.g. float, double)"
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
    vecclip_sse<T>(x.data(), min, max, n);
#elif defined(USE_AVX)
    vecclip_avx<T>(x.data(), min, max, n);
#else
    vecclip_ansi<T>(x, min, max);
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
    has_inf = hasinf_sse<T>(x.data(), n);
#elif defined(USE_AVX)
    has_inf = hasinf_avx<T>(x.data(), n);
#else
    has_inf = hasinf_ansi<T>(x);
#endif
    return has_inf;
};

/**
 * @brief Computes L2 norm of a vector with hardware acceleration
 *        norm2 = Σx²
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
    norm2 = vecnorm2_sse<T>(x.data(), n, squared);
#elif defined(USE_AVX)
    norm2 = vecnorm2_avx<T>(x.data(), n, squared);
#else
    norm2 = vecnorm2_ansi<T>(x, squared);
#endif
    return norm2;
}

/**
 * @brief Computes L2 norm of a memory range [xbegin, xend] with hardware acceleration
 *
 * @tparam T Floating-point type (float/double)
 * @param xbegin Pointer to the first element in the range
 * @param xend Pointer past the last element in the range
 * @param squared When true returns squared norm (skip sqrt)
 *
 * @return T Computed norm value (sqrt(Σx²) when squared=false, Σx² when squared=true)
 *
 * @throws std::invalid_argument For:
 * - Null pointer inputs (xbegin or xend is nullptr)
 * - Invalid range (xbegin >= xend)
 * - Empty input range (xend - xbegin == 0)
 *
 * @note Key implementation details:
 * - Optimized with SSE/AVX vectorization when enabled
 * - Memory range [xbegin, xend) must be contiguous
 * - For best performance with SIMD, ensure memory is 16/32-byte aligned
 *
 * @example
 * float data[] = {1.0f, 2.0f, 3.0f};
 * float norm = vecnorm2(data, data+3, false); // sqrt(14) ≈ 3.7417
 */
template<typename T>
inline T vecnorm2(const T* xbegin,
                  const T* xend,
                  bool squared = false) {
    static_assert(std::is_floating_point_v<T>,
        "vecnorm2 requires floating-point types (e.g. float, double)"
    );

    if (xbegin == nullptr || xend == nullptr) {
        THROW_INVALID_ERROR("vecnorm2: input pointers cannot be null (xbegin="
                            + std::to_string(reinterpret_cast<uintptr_t>(xbegin))
                            + ", xend="
                            + std::to_string(reinterpret_cast<uintptr_t>(xend)) + ")");
    }

    if (xbegin >= xend) {
        THROW_INVALID_ERROR("vecnorm2: invalid range [xbegin, xend) with size="
                            + std::to_string(xend - xbegin));
    }
    const std::size_t n = static_cast<std::size_t>(xend - xbegin);
    if (n == 0) {
        THROW_INVALID_ERROR("vecnorm2: empty input range");
    }

    T norm2;
#if defined(USE_SSE)
    norm2 = vecnorm2_sse<T>(xbegin, n, squared);
#elif defined(USE_AVX)
    norm2 = vecnorm2_avx<T>(xbegin, n, squared);
#else
    norm2 = vecnorm2_ansi<T>(xbegin, xend, squared);
#endif
    return norm2;
}

/**
 * @brief Computes L1 norm of a vector with hardware acceleration
 *        norm1 = Σabs(x)
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
    norm1 = vecnorm1_sse<T>(x.data(), n);
#elif defined(USE_AVX)
    norm1 = vecnorm1_avx<T>(x.data(), n);
#else
    norm1 = vecnorm1_ansi<T>(x);
#endif
    return norm1;
}

/**
 * @brief Scales vector elements by a constant with hardware acceleration
 *        out[i] = x[i] * c
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
    vecscale_sse<T>(x.data(), c, n, out.data());
#elif defined(USE_AVX)
    vecscale_avx<T>(x.data(), c, n, out.data());
#else
    vecscale_ansi<T>(x, c, out);
#endif
}

/**
 * @brief Scales vector elements in a range [xbegin, xend) by a constant with hardware acceleration
 *        out[i] = x[i] * c, i = [begin, end]
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
    vecscale_sse<T>(xbegin, xend, c, n, out.data());
#elif defined(USE_AVX)
    vecscale_avx<T>(xbegin, xend, c, n, out.data());
#else
    vecscale_ansi<T>(xbegin, xend, c, out);
#endif

};

/**
 * @brief Performs vector addition with hardware acceleration
 *        out[i] = x[i] + y[i]
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
    vecadd_sse<T>(x.data(), y.data(), n, m, out.data());
#elif defined(USE_AVX)
    vecadd_avx<T>(x.data(), y.data(), n, m, out.data());
#else
    vecadd_ansi<T>(x, y, out);
#endif
}

/**
 * @brief Performs scaled vector addition with hardware acceleration
 *        out[i] = x[i] * c + y[i]
 * @tparam T Floating-point type (float/double)
 *
 * @param[in] x First input vector
 * @param[in] y Second input vector
 * @param[in] c Scaling factor applied to x
 * @param[out] out Output vector storing element-wise result
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
    vecadd_sse<T>(x.data(), y.data(), c, n, m, out.data());
#elif defined(USE_AVX)
    vecadd_avx<T>(x.data(), y.data(), c, n, m, out.data());
#else
    vecadd_ansi<T>(x, y, c, out);
#endif
}

/**
 * @brief Performs scaled vector accumulation with hardware acceleration
 *
 * Computes the operation: out[i] += x[i] * c for all elements
 *
 * @tparam T Floating-point type (float/double)
 * @param[in] x Input vector to scale and accumulate
 * @param[in] c Scaling factor applied to input elements
 * @param[in,out] out Output vector to accumulate into (must be pre-allocated)
 *
 * @throws std::invalid_argument For:
 * - Empty input/output vectors
 * - Size mismatch between input and output vectors
 *
 * @note Implementation characteristics:
 * - Uses SSE/AVX vectorization when enabled (processes 4/16 elements per cycle)
 * - Memory alignment recommended for optimal performance
 * - Non-atomic operation (not thread-safe for parallel writes)
 *
 * @example
 * std::vector<float> x = {1.0f, 2.0f, 3.0f};
 * std::vector<float> out(3, 0.5f);
 * vecadd(x, 0.5f, out);  // out becomes [1.0f, 1.5f, 2.0f]
 */
template<typename T>
inline void vecadd(const std::vector<T>& x,
                   const T& c,
                   std::vector<T>& out) {

    static_assert(std::is_floating_point_v<T>,
        "vecadd requires floating-point types (e.g. float, double)");

    if (x.empty()) {
        THROW_INVALID_ERROR("vecadd: input x, y vector cannot be empty");
    }

    if (out.empty()) {
        THROW_INVALID_ERROR("vecadd: output vector cannot be empty");
    }

    std::size_t n = x.size();
    if (n != out.size()) {
        THROW_INVALID_ERROR("vecadd: requires x.size() == out.size()");
    }

#if defined(USE_SSE)
    vecadd_sse<T>(x.data(), c, n, out.data());
#elif defined(USE_AVX)
    vecadd_avx<T>(x.data(), c, n, out.data());
#else
    vecadd_ansi<T>(x, c, out);
#endif
}

/**
 *
 */
template<typename T>
inline void vecadd(const T* xbegin,
                   const T* xend,
                   const T& c,
                   std::vector<T>& out) {

    static_assert(std::is_floating_point_v<T>,
        "vecadd requires floating-point types (e.g. float, double)");

    if (xbegin == nullptr || xend == nullptr) {
        THROW_INVALID_ERROR("vecadd: input x, y vector cannot be empty");
    }

    if (xbegin >= xend) {
        THROW_INVALID_ERROR("vecscale: invalid range [xbegin, xend) with size="
                            + std::to_string(xend - xbegin));
    }
    const std::size_t n = static_cast<std::size_t>(xend - xbegin);
    if (n == 0) {
        THROW_INVALID_ERROR("vecscale: empty input range");
    }

    if (out.empty()) {
        THROW_INVALID_ERROR("vecadd: output vector cannot be empty");
    }

#if defined(USE_SSE)
    vecadd_sse<T>(xbegin, c, n, out.data());
#elif defined(USE_AVX)
    vecadd_avx<T>(xbegin, c, n, out.data());
#else
    vecadd_ansi<T>(xbegin, xend, c, out);
#endif
}

/**
 * @brief Performs vector subtraction with hardware acceleration
 *        out[i] = x[i] - y[i]
 *
 * @tparam T Floating-point type (float/double)
 *
 * @param[in] x First input vector
 * @param[in] y Second input vector
 * @param[out] out Output vector storing element-wise difference
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
    vecdiff_sse<T>(x.data(), y.data(), n, m, out.data());
#elif defined(USE_AVX)
    vecdiff_avx<T>(x.data(), y.data(), n, m, out.data());
#else
    vecdiff_ansi<T>(x, y, out);
#endif
}

/**
 * @brief Performs scaled vector subtraction with hardware acceleration
 *        out[i] = x[i] - y[i] * c
 *
 * @tparam T Floating-point type (float/double)
 *
 * @param[in] x First input vector (minuend)
 * @param[in] y Second input vector (subtrahend)
 * @param[in] c Scaling factor applied to y vector elements
 * @param[out] out Output vector storing element-wise
 *
 * @note Implementation priority:
 * 1. AVX vectorization when USE_AVX defined (processes 8 elements per cycle)
 * 2. SSE vectorization when USE_SSE defined (processes 4 elements per cycle)
 * 3. ANSI std::transform otherwise
 *
 * @throws std::invalid_argument For:
 * - Empty input vectors (x or y)
 * - Size mismatch between input vectors (x.size() != y.size())
 * - Output vector size mismatch (out.size() != x.size())
 */
template<typename T>
inline void vecdiff(const std::vector<T>& x,
                    const std::vector<T>& y,
                    const T c,
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
    vecdiff_sse<T>(x.data(), y.data(), c, n, m, out.data());
#elif defined(USE_AVX)
    vecdiff_avx<T>(x.data(), y.data(), c, n, m, out.data());
#else
    vecdiff_ansi<T>(x, y, c, out);
#endif
}

/**
 * @brief Computes dot product of two vectors with hardware acceleration
 *        scalar += x[i] * y[i]
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
    prod = vecdot_sse<T>(x.data(), y.data(), n, m);
#elif defined(USE_AVX)
    prod = vecdot_avx<T>(x.data(), y.data(), n, m);
#else
    prod = vecdot_ansi<T>(x, y);
#endif
    return prod;
}

/**
 *
 */
template<typename T>
inline T vecdot(const T* xbegin,
                const T* xend,
                const T* ybegin) {
    static_assert(std::is_floating_point_v<T>,
        "vecdot requires floating-point types (e.g. float, double)");

    if (xbegin == nullptr || xend == nullptr) {
        THROW_INVALID_ERROR("vecscale: input pointers cannot be null (xbegin="
                            + std::to_string(reinterpret_cast<uintptr_t>(xbegin))
                            + ", xend="
                            + std::to_string(reinterpret_cast<uintptr_t>(xend)) + ")");
    }

    if (ybegin == nullptr) {
        THROW_INVALID_ERROR("vecdot: ybegin pointer cannot be null (ybegin="
                            + std::to_string(reinterpret_cast<uintptr_t>(ybegin)) + ")");
    }

    if (xbegin >= xend) {
        THROW_INVALID_ERROR("vecscale: invalid range [xbegin, xend) with size="
                            + std::to_string(xend - xbegin));
    }
    const std::size_t n = static_cast<std::size_t>(xend - xbegin);
    if (n == 0) {
        THROW_INVALID_ERROR("vecscale: empty input range");
    }
    T prod;
#if defined(USE_SSE)
    prod = vecdot_sse<T>(xbegin, ybegin, n, n);
#elif defined(USE_AVX)
    prod = vecdot_avx<T>(xbegin, ybegin, n, n);
#else
    prod = vecdot_ansi<T>(xbegin, xend, ybegin);
#endif
    return prod;
};

/**
 * @brief Performs element-wise vector multiplication with hardware acceleration
 *        out[i] = x[i] * y[i]
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
    vecmul_sse<T>(x.data(), y.data(), n, m, out.data());
#elif defined(USE_AVX)
    vecmul_avx<T>(x.data(), y.data(), n, m, out.data());
#else
    vecmul_ansi<T>(x, y, out);
#endif
};


/**
 * @brief Computes the accumulated sum of elements in a range [xbegin, xend) with hardware acceleration
 *
 * @tparam T Floating-point type (float/double)
 *
 * @param[in] xbegin Pointer to the first element in the input range
 * @param[in] xend   Pointer past the last element in the input range
 *
 * @return T Sum of all elements in the range [xbegin, xend)
 *
 * @note Implementation priority:
 * 1. AVX vectorization when USE_AVX defined (processes 8 elements per cycle)
 * 2. SSE vectorization when USE_SSE defined (processes 4 elements per cycle)
 * 3. ANSI std::accumulate otherwise
 *
 * @throws std::invalid_argument For:
 * - Null pointer inputs (xbegin or xend is nullptr)
 * - Invalid range (xbegin >= xend)
 * - Empty input range (xend - xbegin == 0)
 */
template<typename T>
inline T vecaccumul(const T* xbegin,
                    const T* xend) {
    static_assert(std::is_floating_point_v<T>,
        "vecdot requires floating-point types (e.g. float, double)");

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
    T prod;
#if defined(USE_SSE)
    prod = vecaccmul_sse<T>(xbegin, xend, n);
#elif defined(USE_AVX)
    prod = vecaccmul_avx<T>(xbegin, xend, n);
#else
    prod = vecaccmul_ansi<T>(xbegin, xend);
#endif
    return prod;
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
 * @brief Computes the row-wise norms of a vector.
 *
 * This function calculates the norms of each row in a given vector `x`,
 * The result can be squared or not, based on the `squared` flag.
 *
 * @tparam T A floating-point or arithmetic type that supports necessary operations.
 * @param x A constant reference to the input ArrayDatasetType vector
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
template<typename T>
void row_norms(const sgdlib::detail::ArrayDatasetType &dataset,
               bool squared,
               std::vector<T> &out) {

    std::size_t nrows = dataset.nrows();
    std::size_t ncols = dataset.ncols();
    std::size_t num_elems = nrows * ncols;
    std::vector<T> xnorm(num_elems);

    // compute x * x
    std::transform(dataset.Xt_data_ptr(),
        dataset.Xt_data_ptr() + num_elems,
        xnorm.begin(), [](const T& x) {
            return x * x;
        }
    );

    // compute prefix sum of xnorm
    std::vector<T> prefix_sum(num_elems);
    std::partial_sum(xnorm.begin(),
        xnorm.end(),
        prefix_sum.begin()
    );

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
        std::transform(out.begin(),
            out.end(), out.begin(),
            [](const T& x) {
                return std::sqrt(x);
            }
        );
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
