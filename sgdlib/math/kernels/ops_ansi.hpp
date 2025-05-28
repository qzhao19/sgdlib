#ifndef MATH_KERNELS_OPS_ANSI_HPP_
#define MATH_KERNELS_OPS_ANSI_HPP_

#include "common/logging.hpp"
#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

template <typename T>
inline void vecset_ansi(std::vector<T>& x, const T c) noexcept {
    std::fill(x.begin(), x.end(), c);
}

template <typename T>
inline void veccpy_ansi(const std::vector<T>& x,
                        std::vector<T>& out) noexcept {
    std::copy(x.begin(), x.end(), out.begin());
}

template <typename T>
inline void vecncpy_ansi(const std::vector<T>& x,
                         std::vector<T>& out) noexcept {
    std::transform(
        x.begin(), x.end(),
        out.begin(),
        [](const T& val) { return -val; }
    );
}

template <typename T>
inline void vecclip_ansi(std::vector<T>& x, T min, T max) noexcept {
    std::transform(
        x.begin(), x.end(), x.begin(),
        [min, max](const T& val) {
            return std::clamp(val, min, max);
        }
    );
};

template <typename T>
inline bool hasinf_ansi(const std::vector<T>& x) noexcept {
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (std::isinf(x[i])) {
            return true;
        }
    }
    return false;
};

template <typename T>
inline T vecnorm2_ansi(const std::vector<T>& x,
                       bool squared) noexcept {
    T l2_norm = std::inner_product(
        x.begin(), x.end(),
        x.begin(),
        static_cast<T>(0)
    );
    return squared ? l2_norm : std::sqrt(l2_norm);
};

template <typename T>
inline T vecnorm2_ansi(const T* xbegin,
                       const T* xend,
                       bool squared) noexcept {
    T l2_norm = std::inner_product(
        xbegin, xend,
        xbegin,
        static_cast<T>(0)
    );
    return squared ? l2_norm : std::sqrt(l2_norm);
};

template <typename T>
inline T vecnorm1_ansi(const std::vector<T>& x) noexcept {
    return std::accumulate(
        x.begin(), x.end(), static_cast<T>(0),
        [](T acc, const T& value) {
            return acc + std::abs(value);
        }
    );
};

template <typename T>
inline void vecscale_ansi(const std::vector<T>& x,
                          const T& c,
                          std::vector<T>& out) noexcept {
    std::transform(
        x.begin(), x.end(),
        out.begin(),
        [&c](const T& val) { return val * c; }
    );
};

template <typename T>
inline void vecscale_ansi(const T* xbegin,
                          const T* xend,
                          const T& c,
                          std::vector<T>& out) noexcept {
    std::transform(
        xbegin, xend,
        out.begin(),
        [&c](const T& val) { return val * c; }
    );
};

template <typename T>
inline void vecadd_ansi(const std::vector<T>& x,
                        const std::vector<T>& y,
                        std::vector<T>& out) noexcept {
    std::transform(
        x.begin(), x.end(),
        y.begin(),
        out.begin(),
        std::plus<T>()
    );
};

template <typename T>
inline void vecadd_ansi(const std::vector<T>& x,
                        const std::vector<T>& y,
                        const T& c,
                        std::vector<T>& out) noexcept {
    std::transform(
        x.begin(), x.end(),
        y.begin(),
        out.begin(),
        [&c](const T& xval, const T& yval) {
            return xval * c + yval;
        }
    );
};

template <typename T>
inline void vecadd_ansi(const std::vector<T>& x,
                        const T c,
                        std::vector<T>& out) {
    std::transform(
        x.begin(), x.end(),
        out.begin(),
        out.begin(),
        [c](const T& xi, const T& outi) {
            return outi + c * xi;  // out[i] += c * x[i]
        }
    );
}

template <typename T>
inline void vecdiff_ansi(const std::vector<T>& x,
                        const std::vector<T>& y,
                        std::vector<T>& out) noexcept {
    std::transform(
        x.begin(), x.end(),
        y.begin(),
        out.begin(),
        std::minus<T>()
    );
};

template <typename T>
inline void vecdiff_ansi(const std::vector<T>& x,
                         const std::vector<T>& y,
                         const T& c,
                         std::vector<T>& out) noexcept {
    std::transform(
        x.begin(), x.end(),
        y.begin(),
        out.begin(),
        [&c](const T& xval, const T& yval) {
            return xval - yval * c;
        }
    );
};

template <typename T>
inline T vecdot_ansi(const std::vector<T>& x,
                     const std::vector<T>& y) noexcept {
    T prod = std::inner_product(
        x.begin(), x.end(),
        y.begin(),
        static_cast<T>(0)
    );
    return prod;
};

template <typename T>
inline T vecdot_ansi(const T* xbegin,
                     const T* xend,
                     const T* ybegin) noexcept {
    T prod = std::inner_product(
        xbegin, xend,
        ybegin,
        static_cast<T>(0)
    );
    return prod;

};

template <typename T>
inline void vecmul_ansi(const std::vector<T>& x,
                        const std::vector<T>& y,
                        std::vector<T>& out) noexcept {
    std::transform(
        x.begin(), x.end(),
        y.begin(),
        out.begin(),
        std::multiplies<T>()
    );
};

template <typename T>
inline T vecaccmul_ansi(const T* xbegin,
                        const T* xend) {
    T acc = std::accumulate(
        xbegin, xend,
        static_cast<T>(0)
    );
    return acc;
};

} // namespace detail
} // namespace sgdlib

#endif // MATH_KERNELS_OPS_ANSI_HPP_
