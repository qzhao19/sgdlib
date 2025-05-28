#ifndef MATH_KERNELS_OPS_AVX_HPP_
#define MATH_KERNELS_OPS_AVX_HPP_

#if defined(USE_AVX)
    #include "ops_avx/ops_avx_float.hpp"
    #include "ops_avx/ops_avx_double.hpp"
#endif

namespace sgdlib {
namespace detail {

template <typename T>
inline void vecset_avx(T* x, const T c, std::size_t n) {
    if constexpr (std::is_same_v<T, float>) {
        vecset_avx_float(x, c, n);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecset_avx_double(x, c, n);
    }
}

template <typename T>
inline void veccpy_avx(const T* x, std::size_t n, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        veccpy_avx_float(x, n, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        veccpy_avx_double(x, n, out);
    }
}

template <typename T>
inline void vecncpy_avx(const T* x, std::size_t n, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecncpy_avx_float(x, n, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecncpy_avx_double(x, n, out);
    }
}

template <typename T>
inline void vecclip_avx(T* x, T min, T max, std::size_t n) {
    if constexpr (std::is_same_v<T, float>) {
        vecclip_avx_float(x, min, max, n);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecclip_avx_double(x, min, max, n);
    }
};

template <typename T>
inline bool hasinf_avx(const T* x, std::size_t n) {
    bool has_inf = false;
    if constexpr (std::is_same_v<T, float>) {
        has_inf = hasinf_avx_float(x, n);
    }
    else if constexpr (std::is_same_v<T, double>) {
        has_inf = hasinf_avx_double(x, n);
    }
    return has_inf;
};

template <typename T>
inline T vecnorm2_avx(const T* x, std::size_t n, bool squared) {
    T norm2;
    if constexpr (std::is_same_v<T, float>) {
        norm2 = vecnorm2_avx_float(x, n, squared);
    }
    else if constexpr (std::is_same_v<T, double>) {
        norm2 = vecnorm2_avx_double(x, n, squared);
    }
    return norm2;
};

template <typename T>
inline T vecnorm1_avx(const T* x, std::size_t n) {
    T norm1;
    if constexpr (std::is_same_v<T, float>) {
        norm1 = vecnorm1_avx_float(x, n);
    }
    else if constexpr (std::is_same_v<T, double>) {
        norm1 = vecnorm1_avx_double(x, n);
    }
    return norm1;
};

template <typename T>
inline void vecscale_avx(const T* x, const T c, std::size_t n, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecscale_avx_float(x, c, n, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecscale_avx_double(x, c, n, out);
    }
};

template <typename T>
inline void vecscale_avx(const T* xbegin, const T* xend, const T c, std::size_t n, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecscale_avx_float(xbegin, xend, c, n, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecscale_avx_double(xbegin, xend, c, n, out);
    }
};

template <typename T>
inline void vecadd_avx(const T* x, const T* y, std::size_t n, std::size_t m, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecadd_avx_float(x, y, n, m, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecadd_avx_double(x, y, n, m, out);
    }
};

template <typename T>
inline void vecadd_avx(const T* x, const T* y, const T c, std::size_t n, std::size_t m, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecadd_avx_float(x, y, c, n, n, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecadd_avx_double(x, y, c, n, n, out);
    }
};

template <typename T>
inline void vecadd_avx(const T* x, const T c, std::size_t n, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecadd_avx_float(x, c, n, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecadd_avx_double(x, c, n, out);
    }
};

template <typename T>
inline void vecdiff_avx(const T* x, const T* y, std::size_t n, std::size_t m, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecdiff_avx_float(x, y, n, m, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecdiff_avx_double(x, y, n, m, out);
    }
};

template <typename T>
inline void vecdiff_avx(const T* x, const T* y, const T c, std::size_t n, std::size_t m, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecdiff_avx_float(x, y, c, n, m, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecdiff_avx_double(x, y, c, n, m, out);
    }
};

template <typename T>
inline T vecdot_avx(const T* x, const T* y, std::size_t n, std::size_t m) {
    T prod;
    if constexpr (std::is_same_v<T, float>) {
        prod = vecdot_avx_float(x, y, n, m);
    }
    else if constexpr (std::is_same_v<T, double>) {
        prod = vecdot_avx_double(x, y, n, m);
    }
    return prod;
};

template <typename T>
inline void vecmul_avx(const T* x, const T* y, std::size_t n, std::size_t m, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        vecmul_avx_float(x, y, n, m, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        vecmul_avx_double(x, y, n, m, out);
    }
};

template <typename T>
inline T vecaccmul_avx(const T* xbegin, const T* xend, std::size_t n) {
    T prod;
    if constexpr (std::is_same_v<T, float>) {
        prod = vecaccuml_avx_float(xbegin, xend, n);
    }
    else if constexpr (std::is_same_v<T, double>) {
        prod = vecaccmul_avx_double(xbegin, xend, n);
    }
    return prod;
};


} // namespace detail
} // namespace sgdlib

#endif // MATH_KERNELS_OPS_AVX_HPP_
