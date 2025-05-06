#ifndef CONTAINER_ALIGNED_VECTOR_HPP_
#define CONTAINER_ALIGNED_VECTOR_HPP_

#include "aligned_allocator.hpp"

namespace sgdlib {

template <typename T, std::size_t Alignment = DEFAULT_MEMORY_ALIGNMENT>
class vector : public std::vector<T, sgdlib::detail::AlignedAllocator<T, Alignment>> {
public:
    using Base = std::vector<T, sgdlib::detail::AlignedAllocator<T, Alignment>>;
    // Inherit all constructors from Base
    using Base::Base;

    // make sure that memory is aligned（optional check）
    bool is_aligned() const noexcept {
        return (reinterpret_cast<std::uintptr_t>(this->data()) & (Alignment - 1)) == 0;
    }

    vector(std::initializer_list<T> init) : Base(init) {}

    template <typename Iter>
    vector(Iter first, Iter last) : Base(first, last) {}

    template <typename T_ = T>
    vector(const T_* first, const T_* last) : Base() {
        static_assert(std::is_same_v<T_, T>, "Pointer type must match vector element type");
        static_assert(std::is_trivially_copyable_v<T_>, "Type must be trivially copyable");
        if (first >= last) return;

        const auto size = static_cast<std::size_t>(last - first);
        // create object with size
        this->resize(size);
        std::memcpy(this->data(), first, size * sizeof(T_));

        // if we use reserve + memcpy
        // this->reserve(size);
        // if (size > 0) {
        //     T* dest = this->data();
        //     std::memcpy(dest, first, size * sizeof(T_));
        //     this->_M_impl._M_finish = this->_M_impl._M_start + size;
        // }

    }
};

}
#endif // CONTAINER_ALIGNED_VECTOR_HPP_
