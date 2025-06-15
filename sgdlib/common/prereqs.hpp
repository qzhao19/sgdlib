#ifndef COMMON_PREREQS_HPP_
#define COMMON_PREREQS_HPP_

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stack>
#include <string>
#include <stdexcept>
#include <typeinfo>
#include <type_traits>
#include <utility>
#include <unordered_set>
#include <vector>

#include <cxxabi.h>

// routine check
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_OPENMP)
namespace sgdlib {
namespace detail {
    inline bool is_sse42_available  = false;
    inline bool is_avx2_available = false;
    inline bool is_openmp_available = false;

    __attribute__((always_inline))
    inline void detect_simd_hardware() noexcept {
        unsigned int max_cpuid, ebx, ecx;

        // bascic CPU info
        __asm__ __volatile__(
            "cpuid" : "=a"(max_cpuid) : "a"(0) : "%ebx", "%ecx", "%edx"
        );

        // SSE4.2 check（CPUID level >= 1）
        if (max_cpuid >= 1) {
            __asm__ __volatile__(
                "cpuid" : "=c"(ecx) : "a"(1) : "%ebx", "%edx"
            );
            is_sse42_available = (ecx & (1 << 20));
        }

        // AVX2 check（CPUID level >= 7）
        if (max_cpuid >= 7) {
            __asm__ __volatile__(
                "cpuid" : "=b"(ebx) : "a"(7), "c"(0) : "%edx"
            );
            is_avx2_available = (ebx & (1 << 5));
        }

        //
        #if defined(USE_AVX)
            if (!is_avx2_available) {
                fprintf(stderr, "[FATAL] Hardware lacks AVX2 support\n");
                std::abort();
            }
        #elif defined(USE_SSE)
            if (!is_sse42_available) {
                fprintf(stderr, "[FATAL] Hardware lacks SSE4.2 support\n");
                std::abort();
            }
        #endif
    };

    __attribute__((always_inline))
    inline void detect_openmp_hardware() noexcept {
        unsigned int max_cpuid, ebx, ecx, edx;

        __asm__ __volatile__(
            "cpuid" : "=a"(max_cpuid) : "a"(0) : "%ebx", "%ecx", "%edx"
        );

        // CPUID level >= 1）
        if (max_cpuid >= 1) {
            __asm__ __volatile__(
                "cpuid" : "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(1)
            );
            is_openmp_available = (edx & (1 << 28)); // HTT bit
        }

        #if defined(_OPENMP)
            if (!is_openmp_available) {
                fprintf(stderr, "[WARNING] Hardware lacks multi-threading support\n");
            }
        #endif
    };

    struct SimdAutoInit {
        SimdAutoInit() { detect_simd_hardware(); }
    };
    inline SimdAutoInit _simd_auto_init;

    struct OpenMPAutoInit {
        OpenMPAutoInit() { detect_openmp_hardware(); }
    };
    inline OpenMPAutoInit _openmp_auto_init;

} // namespace detail
} // namespace sgdlib
#endif

#define HW_SUPPORTS_SSE42() (sgdlib::detail::is_sse42_available)
#define HW_SUPPORTS_AVX2()  (sgdlib::detail::is_avx2_available)
#define HW_SUPPORTS_OPENMP() (sgdlib::detail::is_openmp_available)

// compilation check
#if defined(USE_SSE) || defined(USE_AVX)
// --------------- compiler compatibility check---------------
#if !defined(__GNUC__) || defined(__clang__)
#error "SIMD optimization requires GCC compiler (clang not supported yet)"
#endif

// --------------- if both defined ---------------
#if defined(USE_SSE) && defined(USE_AVX)
#error "USE_SSE and USE_AVX cannot be defined simultaneously"
#endif

// --------------- compiler flag check ---------------
#if defined(USE_SSE) && !defined(__SSE4_2__)
#error "USE_SSE defined but SSE4.2 not enabled. Compile with -msse4.2"
#endif

#if defined(USE_AVX) && !defined(__AVX2__)
#error "USE_AVX defined but AVX2 not enabled. Compile with -mavx2"
#endif
#endif

#if defined(USE_OPENMP)
#ifndef _OPENMP
#error "USE_OPENMP defined but compiler flag -fopenmp not enabled. Please compile with -fopenmp."
#endif
#endif

#if defined(USE_SSE)
#include <smmintrin.h> // SSE4.2
constexpr std::size_t DTYPE_ELEMS_PER_REGISTER = 2; // nums of double stored in register
constexpr std::size_t DTYPE_UNROLLING_FACTOR = 2;   // double-precision expansion factor
constexpr std::size_t DTYPE_UNROLLING_SIZE =
    DTYPE_ELEMS_PER_REGISTER * DTYPE_UNROLLING_FACTOR;
constexpr std::size_t MEMORY_ALIGNMENT = 16;
#elif defined(USE_AVX)
#include <immintrin.h> // AVX2
// float type
constexpr std::size_t FTYPE_ELEMS_PER_REGISTER = 8;  // nums of float stored in register
constexpr std::size_t FTYPE_UNROLLING_FACTOR = 2;    // single-precision expansion factor
constexpr std::size_t FTYPE_UNROLLING_SIZE  =
    FTYPE_ELEMS_PER_REGISTER * FTYPE_UNROLLING_FACTOR; // 16

// double type
constexpr std::size_t DTYPE_ELEMS_PER_REGISTER = 4; // nums of double stored in register
constexpr std::size_t DTYPE_UNROLLING_FACTOR = 4;   // double-precision expansion factor
constexpr std::size_t DTYPE_UNROLLING_SIZE =
    DTYPE_ELEMS_PER_REGISTER * DTYPE_UNROLLING_FACTOR; // 16
constexpr std::size_t MEMORY_ALIGNMENT = 32;
#else
#pragma message("Running in scalar mode (define USE_SSE or USE_AVX for SIMD acceleration)")
#endif

#if defined(USE_OPENMP)
#include <omp.h>
constexpr std::size_t NUM_THREADS = 4;
#endif

// at main or init phase to call this function
inline void check_runtime_simd_openmp() {
#if defined(USE_SSE)
    if (!HW_SUPPORTS_SSE42()) {
        fprintf(stderr, "[FATAL] SSE4.2 not supported at runtime\n");
        std::abort();
    }
#elif defined(USE_AVX)
    if (!HW_SUPPORTS_AVX2()) {
        fprintf(stderr, "[FATAL] AVX2 not supported at runtime\n");
        std::abort();
    }
#endif
#if defined(USE_OPENMP)
    if (!HW_SUPPORTS_OPENMP()) {
        fprintf(stderr, "[FATAL] OpenMP not supported at runtime\n");
        std::abort();
    }
#endif
}

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

#endif // COMMON_PREREQS_HPP_
