#ifndef MATH_KERNELS_OPS_SSE_HPP_
#define MATH_KERNELS_OPS_SSE_HPP_

#include "ops_sse/ops_sse_double.hpp"
#include "ops_sse/ops_sse_float.hpp"

namespace sgdlib {
namespace detail {

template <typename T>
void vecset_sse(T* x, const T c, std::size_t n) {
    if constexpr (std::is_same_v<T, float>) {
        vecset_sse_float(x, c, n);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecset_sse_double(x, c, n);
    }
}

template <typename T>
void veccpy_sse(const T* x, std::size_t n, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        veccpy_sse_float(x, n, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        veccpy_sse_double(x, n, out);
    }
}

template <typename T>
void vecncpy_sse(const T* x, std::size_t n, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecncpy_sse_float(x, n, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecncpy_sse_double(x, n, out);
    }
}

template <typename T>
void vecclip_sse(T* x, T min, T max, std::size_t n) {
    if constexpr (std::is_same_v<T, float>) {
        vecclip_sse_float(x, min, max, n);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecclip_sse_double(x, min, max, n);
    }
};

template <typename T>
bool hasinf_sse(const T* x, std::size_t n) {
    bool retval;
    if constexpr (std::is_same_v<T, float>) {
        retval = hasinf_sse_float(x, n);
    }
    else if constexpr (std::is_same_v<T, double>) {
        retval = hasinf_sse_double(x, n);
    }
    return retval;
};

template <typename T>
T vecnorm2_sse(const T* x, std::size_t n, bool squared) {
    T retval;
    if constexpr (std::is_same_v<T, float>) {
        retval = vecnorm2_sse_float(x, n, squared);
    }
    else if constexpr (std::is_same_v<T, double>) {
        retval = vecnorm2_sse_double(x, n, squared);
    }
    return retval;
};

template <typename T>
T vecnorm1_sse(const T* x, std::size_t n) {
    T retval;
    if constexpr (std::is_same_v<T, float>) {
        retval = vecnorm1_sse_float(x, n);
    }
    else if constexpr (std::is_same_v<T, double>) {
        retval = vecnorm1_sse_double(x, n);
    }
    return retval;
};

template <typename T>
void vecscale_sse(const T* x, const T c, std::size_t n, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecscale_sse_float(x, c, n, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecscale_sse_double(x, c, n, out);
    }
};

template <typename T>
void vecscale_sse(const T* xbegin, const T* xend, const T c, std::size_t n, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecscale_sse_float(xbegin, xend, c, n, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecscale_sse_double(xbegin, xend, c, n, out);
    }
};

template <typename T>
void vecadd_sse(const T* x, const T* y, std::size_t n, std::size_t m, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecadd_sse_float(x, y, n, m, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecadd_sse_double(x, y, n, m, out);
    }
};

template <typename T>
void vecadd_sse(const T* x, const T* y, const T c, std::size_t n, std::size_t m, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecadd_sse_float(x, y, c, n, n, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecadd_sse_double(x, y, c, n, n, out);
    }
};

template <typename T>
void vecdiff_sse(const T* x, const T* y, std::size_t n, std::size_t m, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecdiff_sse_float(x, y, n, m, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecdiff_sse_double(x, y, n, m, out);
    }
};

template <typename T>
T vecdot_sse(const T* x, const T* y, std::size_t n, std::size_t m) {
    T retval;
    if constexpr (std::is_same_v<T, float>) {
        retval = vecdot_sse_float(x, y, n, m);
    }
    else if constexpr (std::is_same_v<T, double>) {
        retval = vecdot_sse_double(x, y, n, m);
    }
    return retval;
};

template<typename T>
void vecmul_sse(const T* x, const T* y, std::size_t n, std::size_t m, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecmul_sse_float(x, y, n, m, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecmul_sse_double(x, y, n, m, out);
    }
};

} // namespace detail
} // namespace sgdlib

#endif // MATH_KERNELS_OPS_SSE_HPP_
