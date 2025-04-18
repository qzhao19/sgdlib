#ifndef COMMON_PREREQS_HPP_
#define COMMON_PREREQS_HPP_

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
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

#if defined(USE_SIMD)
    #if !defined(__GNUC__) || defined(__clang__)
        #error "SIMD optimization requires GCC compiler (clang not supported)"
    #endif

    // setup SSE4.1
    #if !defined(__SSE4_2__)
        #error "SSE4.2 is required when USE_SIMD is defined. Compile with -msse4.2"
    #else
        #include <smmintrin.h>
        #define USE_SSE 1
    #endif

    // check AVX2（optional）
    #if defined(__AVX2__)
        #include <immintrin.h>
        #define USE_AVX2 1
    #else
        #define USE_AVX2 0
        #pragma message("AVX2 not available (will use SSE4.2 instead)")
    #endif

    #if USE_AVX2 && USE_SSE
        #error "Both AVX2 and SSE4.1 are defined. Only one can be used at a time."
    #endif

    // define the active SIMD level
    // Case 1: use SIMD（AVX2 level 2，SSE4.1 level 1)
    #if USE_AVX2
        #define SIMD_TARGET_LEVEL 2  // AVX2
    #else
        #define SIMD_TARGET_LEVEL 1  // SSE4.2
    #endif

// Case 2: use scalar
#else
    #define SIMD_TARGET_LEVEL 0
    #define USE_SSE 0
    #define USE_AVX2 0
#endif // USE_SIMD

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

#endif // COMMON_PREREQS_HPP_
