#ifndef CONTAINER_ALIGNED_ALLOCATOR_HPP_
#define CONTAINER_ALIGNED_ALLOCATOR_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

template<typename T, std::size_t Alignment = DEFAULT_MEMORY_ALIGNMENT>
struct AlignedAllocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    constexpr AlignedAllocator() noexcept = default;

    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    // allocate aligned memory
    [[nodiscard]] pointer allocate(size_type n) {
        // avoid overflow
        if (n > std::numeric_limits<size_type>::max() / sizeof(value_type)) {
            throw std::bad_alloc();
        }

        // allocate aligned memory
        void* ptr = std::aligned_alloc(Alignment, n * sizeof(value_type));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(ptr);
    }

    // free memory
    void deallocate(pointer ptr, size_type) {
        std::free(ptr);
    }

    template <typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
        return true;
    }

    template <typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept {
        return false;
    }
};

}
}


#endif // CONTAINER_ALIGNED_ALLOCATOR_HPP_
