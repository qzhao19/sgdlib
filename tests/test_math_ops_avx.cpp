#include <gtest/gtest.h>
#include <chrono>

#include "sgdlib/common/prereqs.hpp"
#include "sgdlib/container/aligned_vector.hpp"
#include "sgdlib/math/kernels/ops_ansi.hpp"
#include "sgdlib/math/kernels/ops_avx.hpp"

// ****************************generate testing data*******************************//
template<typename T>
class MathOpsAVXTest : public ::testing::Test {
protected:
    using Type = T;

    void SetUp() override {
        engine_.seed(42);
        aligned_mem_vec = static_cast<T*>(_mm_malloc(1024 * sizeof(T), 32));
        unaligned_mem_vec = static_cast<T*>(malloc(1024 * sizeof(T) + 31)) + 1;
        aligned_mem_vec2 = static_cast<T*>(_mm_malloc(1024 * sizeof(T), 32));
        unaligned_mem_vec2 = static_cast<T*>(malloc(1024 * sizeof(T) + 31)) + 1;
    }

    void TearDown() override {
        _mm_free(aligned_mem_vec);
        free(unaligned_mem_vec - 1);
        _mm_free(aligned_mem_vec2);
        free(unaligned_mem_vec2 - 1);
    }

    sgdlib::vector<T> generate_test_data(std::size_t size,
                                      bool with_inf,
                                      T max = std::numeric_limits<T>::max() / 2,
                                      T min = -std::numeric_limits<T>::max() / 2) {
        std::uniform_real_distribution<T> dist(min, max);
        sgdlib::vector<T> arr(size);
        for (auto& val : arr) {
            val = dist(engine_);
        }
        if (with_inf) {
            std::uniform_int_distribution<std::size_t> pos_dist(0, size-1);
            arr[pos_dist(engine_)] = std::numeric_limits<T>::infinity();
        }
        return arr;
    }
    std::mt19937 engine_;
    T* aligned_mem_vec = nullptr;
    T* aligned_mem_vec2 = nullptr;
    T* unaligned_mem_vec = nullptr;
    T* unaligned_mem_vec2 = nullptr;
};

using TestValueTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MathOpsAVXTest, TestValueTypes);


// ****************************vecset*******************************//
TYPED_TEST(MathOpsAVXTest, VecSetAVXTest) {
    using T = typename TestFixture::Type;

    // empty pointer and n = 0
    sgdlib::detail::vecset_avx<T>(nullptr, 1.0, 10);
    sgdlib::detail::vecset_avx<T>(this->aligned_mem_vec, 1.0, 10);

    // small size n < 4
    sgdlib::vector<T> data11(3);
    sgdlib::vector<T> data12(3);
    sgdlib::detail::vecset_avx<T>(data11.data(), 1.0, 3);
    sgdlib::detail::vecset_ansi<T>(data12, 1.0);
    for (std::size_t i = 3; i < 3; ++i) {
        EXPECT_FLOAT_EQ(data11[i], data12[i]);
    }

    // aligned case n = 32
    const std::size_t size2 = 32;
    sgdlib::vector<T> data21(size2);
    sgdlib::vector<T> data22(size2);
    sgdlib::detail::vecset_avx<T>(data21.data(), 1.0, size2);
    sgdlib::detail::vecset_ansi<T>(data22, 1.0);

    for (std::size_t i = size2; i < size2; ++i) {
        EXPECT_FLOAT_EQ(data11[i], data12[i]);
    }

    // big memory block
    const std::size_t size4 = 1024;
    const T c4 = 42.0;
    std::fill_n(this->aligned_mem_vec, size4, c4);

    sgdlib::detail::vecset_avx<T>(this->aligned_mem_vec, c4, size4);
    for (std::size_t i = 0; i < size4; i += 1 + i % 7) {
        EXPECT_FLOAT_EQ(this->aligned_mem_vec[i], c4) << "Failed at i=" << i;
    }

    // remaing elements case
    const std::size_t size5 = 19;
    sgdlib::vector<T> data5(size5);

    sgdlib::detail::vecset_avx<T>(data5.data(), c4, size5);
    for (std::size_t i = 0; i < size5; ++i) {
        EXPECT_FLOAT_EQ(data5[i], c4) << "Failed at i=" << i;
    }

    // large size n = 1 << 20
    const std::size_t size6 = 1 << 20;
    const T c6 = 666;
    sgdlib::vector<T> large_vec;
    large_vec.reserve(size6);
    large_vec.resize(size6);

    auto start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecset_avx<T>(large_vec.data(), c6, size6);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecset SIMD AVX execution time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecset_ansi<T>(large_vec, c6);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "vecset ANSI execution time: " << elapsed2.count() << " seconds\n";
    std::cout << "Speedup: " << elapsed2.count() / elapsed1.count() << "x\n";

    for (std::size_t i = 0; i < size6; ++i) {
        EXPECT_FLOAT_EQ(large_vec[i], c6) << "Failed at i=" << i;
    }
}

// ****************************veccpy*******************************//
TYPED_TEST(MathOpsAVXTest, VecCopyAvxTest) {
    using T = typename TestFixture::Type;

    // should not failed
    sgdlib::vector<T> src1(1024);
    sgdlib::vector<T> dst1(1024);

    sgdlib::detail::veccpy_avx<T>(src1.data(), dst1.data(), 0);
    sgdlib::detail::veccpy_avx<T>(nullptr, dst1.data(), 1024);
    sgdlib::detail::veccpy_avx<T>(src1.data(), nullptr, 1025);
    sgdlib::detail::veccpy_avx<T>(nullptr, nullptr, 1024);

    // n < 8 n < 4
    sgdlib::vector<T> src21(7, 0);
    sgdlib::vector<T> dst21(7, 0);
    sgdlib::vector<T> src22(3, 0);
    sgdlib::vector<T> dst22(3, 0);
    sgdlib::detail::veccpy_avx<T>(src21.data(), dst21.data(), 7);
    sgdlib::detail::veccpy_avx<T>(src22.data(), dst22.data(), 3);

    for (size_t i = 0; i < 7; ++i) {
        EXPECT_FLOAT_EQ(src21[i], dst21[i])
            << "Mismatch at index " << i << " for size " << 3;
    }
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(src22[i], dst22[i])
            << "Mismatch at index " << i << " for size " << 3;
    }


    // // m >= 4
    sgdlib::vector<T> src3(16, 10);
    sgdlib::vector<T> dst3(16, 0);
    sgdlib::detail::veccpy_avx<T>(src3.data(), dst3.data(), 16);
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(src3[i], dst3[i])
            << "Mismatch at index " << i << " for size " << 16;
    }

    // // aligned memeory
    std::fill_n(this->aligned_mem_vec, 36, 2.0);
    sgdlib::detail::veccpy_avx<T>(this->aligned_mem_vec, this->aligned_mem_vec2, 36);
    for (size_t i = 0; i < 36; ++i) {
        EXPECT_FLOAT_EQ(this->aligned_mem_vec[i], this->aligned_mem_vec2[i])
            << "Mismatch at index " << i << " for size " << 36;
    }

    // differnent size
    sgdlib::vector<T> src4(1024, 0);
    sgdlib::vector<T> dst4(1024, 0);

    for (size_t n : {4, 7, 15, 31, 63, 127, 255, 511, 1023}) {
        sgdlib::detail::veccpy_avx<T>(src4.data(), dst4.data(), n);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_FLOAT_EQ(src4[i], dst4[i])
                << "Mismatch at index " << i << " for size " << n;
        }
    }

    // performance test with large size
    const std::size_t size5 = 1 << 20;
    sgdlib::vector<T> large_src1, large_dst1;
    large_src1.reserve(size5);
    large_src1.resize(size5);
    large_dst1.reserve(size5);
    large_dst1.resize(size5);

    auto start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::veccpy_avx<T>(large_src1.data(), large_dst1.data(), size5);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "veccpy SIMD execution time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::veccpy_ansi<T>(large_src1, large_dst1);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "veccpy ANSI execution time: " << elapsed2.count() << " seconds\n";
    std::cout << "Speedup: " << elapsed2.count() / elapsed1.count() << "x\n";

    for (size_t i = 0; i < size5; ++i) {
        EXPECT_FLOAT_EQ(large_src1[i], large_dst1[i])
            << "Mismatch at index " << i << " for size " << size5;
    }
}


// ****************************vecncpy*******************************//
TYPED_TEST(MathOpsAVXTest, VecNegCopyAVXTest) {
    using T = typename TestFixture::Type;

    // should not failed
    sgdlib::vector<T> src1(1024);
    sgdlib::vector<T> dst1(1024);

    sgdlib::detail::vecncpy_avx<T>(src1.data(), dst1.data(), 0);
    sgdlib::detail::vecncpy_avx<T>(nullptr, dst1.data(), 1024);
    sgdlib::detail::vecncpy_avx<T>(src1.data(), nullptr, 1025);
    sgdlib::detail::vecncpy_avx<T>(nullptr, nullptr, 1024);

    // n < 8 n < 4
    sgdlib::vector<T> src21(7, 1);
    sgdlib::vector<T> dst21(7, 1);
    sgdlib::vector<T> src22(3, 1);
    sgdlib::vector<T> dst22(3, 1);
    sgdlib::detail::vecncpy_avx<T>(src21.data(), dst21.data(), 7);
    sgdlib::detail::vecncpy_avx<T>(src22.data(), dst22.data(), 3);

    for (size_t i = 0; i < 7; ++i) {
        EXPECT_FLOAT_EQ(src21[i], -dst21[i])
            << "Mismatch at index " << i << " for size " << 3;
    }
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(src22[i], -dst22[i])
            << "Mismatch at index " << i << " for size " << 3;
    }

    // m >= 4
    sgdlib::vector<T> src3(16, 2);
    sgdlib::vector<T> dst3(16, 2);
    sgdlib::detail::vecncpy_avx<T>(src3.data(), dst3.data(), 16);
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(src3[i], -dst3[i])
            << "Mismatch at index " << i << " for size " << 16;
    }

    // aligned memeory
    std::fill_n(this->aligned_mem_vec, 45, 1.0);
    sgdlib::detail::vecncpy_avx<T>(this->aligned_mem_vec, this->aligned_mem_vec2, 45);
    for (size_t i = 0; i < 45; ++i) {
        EXPECT_FLOAT_EQ(this->aligned_mem_vec[i], -this->aligned_mem_vec2[i])
            << "Mismatch at index " << i << " for size " << 45;
    }

    // differnent size
    sgdlib::vector<T> src4(1024, 2);
    sgdlib::vector<T> dst4(1024, 2);

    for (size_t n : {4, 7, 15, 31, 63, 127, 255, 511, 1023}) {
        sgdlib::detail::vecncpy_avx<T>(src4.data(), dst4.data(), n);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_FLOAT_EQ(src4[i], -dst4[i])
                << "Mismatch at index " << i << " for size " << n;
        }
    }

    // performance test with large size
    const std::size_t size5 = 1 << 20;
    sgdlib::vector<T> large_src1, large_dst1;
    large_src1.reserve(size5);
    large_src1.resize(size5);
    large_dst1.reserve(size5);
    large_dst1.resize(size5);
    std::fill(large_src1.begin(), large_src1.end(), 2);

    auto start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecncpy_avx<T>(large_src1.data(), large_dst1.data(), size5);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecncpy SIMD execution time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecncpy_ansi<T>(large_src1, large_dst1);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "vecncpy ANSI execution time: " << elapsed2.count() << " seconds\n";
    std::cout << "Speedup: " << elapsed2.count() / elapsed1.count() << "x\n";

    for (size_t i = 0; i < size5; ++i) {
        EXPECT_FLOAT_EQ(large_src1[i], -large_dst1[i])
            << "Mismatch at index " << i << " for size " << size5;
    }
}


// ****************************vecclip*******************************//
TYPED_TEST(MathOpsAVXTest, VecClipAVXTest) {
    using T = typename TestFixture::Type;

    // empty vec
    sgdlib::detail::vecclip_avx<T>(nullptr, 0.0, 2.0, 10);
    sgdlib::detail::vecclip_avx<T>(this->aligned_mem_vec, 0.0, 2.0, 0);

    // Test n < 7 cases
    sgdlib::vector<T> data1 = {1.5};
    sgdlib::vector<T> expected1 = data1;
    sgdlib::detail::vecclip_ansi<T>(expected1, 0.0, 2.0);
    sgdlib::detail::vecclip_avx<T>(data1.data(), 0.0, 2.0, data1.size());
    EXPECT_EQ(data1, expected1);

    sgdlib::vector<T> data2 = {-1.0, 3.0, 0.5, 1.2, 0.8, 0.9, 1.0};
    sgdlib::vector<T> expected2 = data2;
    sgdlib::detail::vecclip_ansi<T>(expected2, 0.0, 1.0);
    sgdlib::detail::vecclip_avx<T>(data2.data(), 0.0, 1.0, data2.size());
    EXPECT_EQ(data2, expected2);

    // Test n = 8 (exactly one SIMD operation)
    sgdlib::vector<T> data3 = {-2.0, 0.5, 1.5, 3.0, 1.2, 0.8, 0.9, 1.0};
    sgdlib::vector<T> expected3 = data3;
    sgdlib::detail::vecclip_ansi<T>(expected3, 0.0, 1.0);
    sgdlib::detail::vecclip_avx<T>(data3.data(), 0.0, 1.0, data3.size());
    EXPECT_EQ(data3, expected3);

    // prime number size
    sgdlib::vector<T> data4(127, 0.0);
    for (std::size_t i = 0; i < data4.size(); ++i) {
        data4[i] = static_cast<T>(i) - 50.0;
    }

    sgdlib::vector<T> expected4 = data4;
    sgdlib::detail::vecclip_ansi<T>(expected4, -10.0, 10.0);
    sgdlib::detail::vecclip_avx<T>(data4.data(), -10.0, 10.0, data4.size());

    for (std::size_t i = 0; i < data4.size(); ++i) {
        EXPECT_FLOAT_EQ(data4[i], expected4[i]) << "Mismatch at index " << i;
    }

    // big size n = 1024
    const std::size_t size4 = 1024;
    sgdlib::vector<T> aligned_mem_vec_expected4(size4);
    std::fill_n(this->aligned_mem_vec, size4, 0.0);
    for (std::size_t i = 0; i < size4; ++i) {
        this->aligned_mem_vec[i] = static_cast<T>(i) - 50.0;
        aligned_mem_vec_expected4[i] = static_cast<T>(i) - 50.0;
    }

    sgdlib::detail::vecclip_ansi<T>(aligned_mem_vec_expected4, -10.0, 10.0);
    sgdlib::detail::vecclip_avx<T>(this->aligned_mem_vec, -10.0, 10.0, size4);

    for (std::size_t i = 0; i < size4; ++i) {
        EXPECT_FLOAT_EQ(this->aligned_mem_vec[i], aligned_mem_vec_expected4[i]) << "Mismatch at index " << i;
    }

    // Test NaN/inf (behavior depends on requirements)
    sgdlib::vector<T> specials = {NAN, INFINITY, -INFINITY};
    sgdlib::vector<T> spec_expected = specials;
    sgdlib::detail::vecclip_ansi<T>(spec_expected, 0.0, 1.0);
    sgdlib::detail::vecclip_avx<T>(specials.data(), 0.0, 1.0, specials.size());
    for (std::size_t i = 0; i < specials.size(); ++i) {
        EXPECT_EQ(std::isnan(specials[i]), std::isnan(spec_expected[i]));
        if (!std::isnan(specials[i])) {
            EXPECT_FLOAT_EQ(specials[i], spec_expected[i]);
        }
    }

    // large size n = 1 << 20
    const std::size_t size5 = 1 << 20;
    sgdlib::vector<T> large_vec1, large_vec2;
    large_vec1.reserve(size5);
    large_vec1.resize(size5);
    large_vec2.reserve(size5);
    large_vec2.resize(size5);
    // std::fill_n(large_vec.data(), size5, 0.0);
    for (std::size_t i = 0; i < size5; ++i) {
        large_vec1[i] = static_cast<T>(i) - 50.0;
        large_vec2[i] = static_cast<T>(i) - 50.0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecclip_avx<T>(large_vec1.data(), -10.0, 10.0, size5);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecclip SIMD execution time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecclip_ansi<T>(large_vec2, -10.0, 10.0);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "vecclip ANSI execution time: " << elapsed2.count() << " seconds\n";
    std::cout << "Speedup: " << elapsed2.count() / elapsed1.count() << "x\n";
}


// *****************************hasinf*******************************//
TYPED_TEST(MathOpsAVXTest, VecHasInfAVXTest) {
    using T = typename TestFixture::Type;

    T* data1 = nullptr;
    EXPECT_FALSE(sgdlib::detail::hasinf_avx<T>(data1, 0));

    // // small size array
    sgdlib::vector<T> data2 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    EXPECT_FALSE(sgdlib::detail::hasinf_avx<T>(data2.data(), data2.size()));
    sgdlib::vector<T> data21 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, std::numeric_limits<T>::infinity()};
    EXPECT_TRUE(sgdlib::detail::hasinf_avx<T>(data21.data(), data21.size()));
    sgdlib::vector<T> data22 = {1.0, 2.0, 3.0};
    EXPECT_FALSE(sgdlib::detail::hasinf_avx<T>(data22.data(), data22.size()));

    // aligned size array
    constexpr std::size_t size3 = 128;
    auto data3 = this->generate_test_data(size3, false);
    EXPECT_FALSE(sgdlib::detail::hasinf_avx<T>(data3.data(), size3));

    data3.back() = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(sgdlib::detail::hasinf_avx<T>(data3.data(), size3));

    // unaligned size array
    constexpr std::size_t size4 = 127;
    auto data4 = this->generate_test_data(size4, false);
    EXPECT_FALSE(sgdlib::detail::hasinf_avx<T>(data4.data(), size4));

    data4[125] = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(sgdlib::detail::hasinf_avx<T>(data4.data(), size4));

    // aligned memory array
    std::fill_n(this->aligned_mem_vec, size4, 0.0);
    EXPECT_FALSE(sgdlib::detail::hasinf_avx<T>(this->aligned_mem_vec, size4));

    this->aligned_mem_vec[125] = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(sgdlib::detail::hasinf_avx<T>(this->aligned_mem_vec, size4));

    // large size array
    constexpr std::size_t size5 = 1'000'000;
    // without inf
    auto data5 = this->generate_test_data(size5, false);
    EXPECT_FALSE(sgdlib::detail::hasinf_avx<T>(data5.data(), size5));

    // with inf
    auto data51 = this->generate_test_data(size5, true);
    EXPECT_TRUE(sgdlib::detail::hasinf_avx<T>(data51.data(), size5));

    // edge cases
    sgdlib::vector<T> data6 = {
        std::numeric_limits<T>::min(),
        std::numeric_limits<T>::max(),
        -std::numeric_limits<T>::max()
    };
    EXPECT_FALSE(sgdlib::detail::hasinf_avx<T>(data6.data(), data6.size()));

}


