#include <gtest/gtest.h>
#include <chrono>

#include "sgdlib/common/prereqs.hpp"
#include "sgdlib/math/kernels/ops_ansi.hpp"
#include "sgdlib/math/kernels/ops_sse.hpp"

// ****************************generate testing data*******************************//
template<typename T>
class MathOpsSSETest : public ::testing::Test {
protected:
    using Type = T;

    void SetUp() override {
        engine_.seed(42);
        aligned_mem_vec = static_cast<T*>(_mm_malloc(1024 * sizeof(T), 16));
        unaligned_mem_vec = static_cast<T*>(malloc(1024 * sizeof(T) + 15)) + 1;
        aligned_mem_vec2 = static_cast<T*>(_mm_malloc(1024 * sizeof(T), 16));
        unaligned_mem_vec2 = static_cast<T*>(malloc(1024 * sizeof(T) + 15)) + 1;
    }

    void TearDown() override {
        _mm_free(aligned_mem_vec);
        free(unaligned_mem_vec - 1);
        _mm_free(aligned_mem_vec2);
        free(unaligned_mem_vec2 - 1);
    }

    std::vector<T> generate_test_data(std::size_t size,
                                      bool with_inf,
                                      T max = std::numeric_limits<T>::max() / 2,
                                      T min = -std::numeric_limits<T>::max() / 2) {
        std::uniform_real_distribution<T> dist(min, max);
        std::vector<T> arr(size);
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
TYPED_TEST_SUITE(MathOpsSSETest, TestValueTypes);


// ****************************vecset*******************************//
TYPED_TEST(MathOpsSSETest, VecSetSSETest) {
    using T = typename TestFixture::Type;

    // empty pointer and n = 0
    sgdlib::detail::vecset_sse<T>(nullptr, 1.0, 10);
    sgdlib::detail::vecset_sse<T>(this->aligned_mem_vec, 1.0, 10);

    // small size n < 4
    std::vector<T> data11(3);
    std::vector<T> data12(3);
    sgdlib::detail::vecset_sse<T>(data11.data(), 1.0, 3);
    sgdlib::detail::vecset_ansi<T>(data12, 1.0);
    for (std::size_t i = 3; i < 3; ++i) {
        EXPECT_FLOAT_EQ(data11[i], data12[i]);
    }

    // aligned case n = 32
    const std::size_t size2 = 32;
    std::vector<T> data21(size2);
    std::vector<T> data22(size2);
    sgdlib::detail::vecset_sse<T>(data21.data(), 1.0, size2);
    sgdlib::detail::vecset_ansi<T>(data22, 1.0);

    for (std::size_t i = size2; i < size2; ++i) {
        EXPECT_FLOAT_EQ(data11[i], data12[i]);
    }

    // unaligned memeory
    const std::size_t size3 = 1021;
    std::fill_n(this->unaligned_mem_vec, size3, 0.0);
    sgdlib::detail::vecset_sse<T>(this->unaligned_mem_vec, 1.5, size3);
    for (std::size_t i = 0; i < size3; ++i) {
        EXPECT_FLOAT_EQ(this->unaligned_mem_vec[i], 1.5) << "Failed at i=" << i;
    }

    // aligned memory block
    const std::size_t size4 = 1024;
    const T c4 = 42.0;
    std::fill_n(this->aligned_mem_vec, size4, c4);

    sgdlib::detail::vecset_sse<T>(this->aligned_mem_vec, c4, size4);
    for (std::size_t i = 0; i < size4; i += 1 + i % 7) {
        EXPECT_FLOAT_EQ(this->aligned_mem_vec[i], c4) << "Failed at i=" << i;
    }

    // remaing elements case
    const std::size_t size5 = 19;
    std::vector<T> data5(size5);

    sgdlib::detail::vecset_sse<T>(data5.data(), c4, size5);
    for (std::size_t i = 0; i < size5; ++i) {
        EXPECT_FLOAT_EQ(data5[i], c4) << "Failed at i=" << i;
    }

    // large size n = 1 << 20
    const std::size_t size6 = 1 << 20;
    const T c6 = 666;
    std::vector<T> large_vec;
    large_vec.reserve(size6);
    large_vec.resize(size6);

    auto start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecset_sse<T>(large_vec.data(), c6, size6);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecset SIMD execution time: " << elapsed1.count() << " seconds\n";

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
TYPED_TEST(MathOpsSSETest, VecCopySSETest) {
    using T = typename TestFixture::Type;

    // should not failed
    std::vector<T> src1(1024);
    std::vector<T> dst1(1024);

    sgdlib::detail::veccpy_sse<T>(src1.data(), 0, dst1.data());
    sgdlib::detail::veccpy_sse<T>(nullptr, 1024, dst1.data());
    sgdlib::detail::veccpy_sse<T>(src1.data(), 1025, nullptr);
    sgdlib::detail::veccpy_sse<T>(nullptr, 1024, nullptr);

    // n < 2 n < 4
    std::vector<T> src21(1, 0);
    std::vector<T> dst21(1, 0);
    std::vector<T> src22(3, 0);
    std::vector<T> dst22(3, 0);
    sgdlib::detail::veccpy_sse<T>(src21.data(), 1, dst21.data());
    sgdlib::detail::veccpy_sse<T>(src22.data(), 3, dst22.data());

    EXPECT_FLOAT_EQ(src21[0], dst21[0]) << "Mismatch at index " << 0 << " for size " << 1;
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(src22[i], dst22[i])
            << "Mismatch at index " << i << " for size " << 3;
    }


    // m >= 4
    std::vector<T> src3(16, 0);
    std::vector<T> dst3(16, 0);
    sgdlib::detail::veccpy_sse<T>(src3.data(), 16, dst3.data());
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(src3[i], dst3[i])
            << "Mismatch at index " << i << " for size " << 16;
    }

    // unaligned memory
    std::fill_n(this->unaligned_mem_vec, 31, 1.5);
    sgdlib::detail::veccpy_sse<T>(this->unaligned_mem_vec, 32, this->unaligned_mem_vec2);
    for (size_t i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(this->unaligned_mem_vec[i], this->unaligned_mem_vec2[i])
            << "Mismatch at index " << i << " for size " << 32;
    }

    // aligned memeory
    std::fill_n(this->aligned_mem_vec, 32, 1.0);
    sgdlib::detail::veccpy_sse<T>(this->aligned_mem_vec, 32, this->aligned_mem_vec2);
    for (size_t i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(this->aligned_mem_vec[i], this->aligned_mem_vec2[i])
            << "Mismatch at index " << i << " for size " << 32;
    }

    // differnent size
    std::vector<T> src4(1024, 0);
    std::vector<T> dst4(1024, 0);
    for (size_t n : {4, 7, 15, 31, 63, 127, 255, 511, 1023}) {
        sgdlib::detail::veccpy_sse<T>(src4.data(), n, dst4.data());
        for (size_t i = 0; i < n; ++i) {
            EXPECT_FLOAT_EQ(src4[i], dst4[i])
                << "Mismatch at index " << i << " for size " << n;
        }
    }

    // performance test with large size
    const std::size_t size5 = 1 << 20;
    std::vector<T> large_src1, large_dst1;
    large_src1.reserve(size5);
    large_src1.resize(size5);
    large_dst1.reserve(size5);
    large_dst1.resize(size5);

    auto start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::veccpy_sse<T>(large_src1.data(), size5, large_dst1.data());
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
TYPED_TEST(MathOpsSSETest, VecNegCopySSETest) {
    using T = typename TestFixture::Type;

    // should not failed
    std::vector<T> src1(1024);
    std::vector<T> dst1(1024);

    sgdlib::detail::vecncpy_sse<T>(src1.data(), 0, dst1.data());
    sgdlib::detail::vecncpy_sse<T>(nullptr, 1024, dst1.data());
    sgdlib::detail::vecncpy_sse<T>(src1.data(), 1025, nullptr);
    sgdlib::detail::vecncpy_sse<T>(nullptr, 1024, nullptr);

    // n < 2 n < 4
    std::vector<T> src21(1, 1);
    std::vector<T> dst21(1, 1);
    std::vector<T> src22(3, 1);
    std::vector<T> dst22(3, 1);
    sgdlib::detail::vecncpy_sse<T>(src21.data(), 1, dst21.data());
    sgdlib::detail::vecncpy_sse<T>(src22.data(), 3, dst22.data());

    EXPECT_FLOAT_EQ(src21[0], -dst21[0]) << "Mismatch at index " << 0 << " for size " << 1;
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(src22[i], -dst22[i])
            << "Mismatch at index " << i << " for size " << 3;
    }

    // m >= 4
    std::vector<T> src3(16, 2);
    std::vector<T> dst3(16, 2);
    sgdlib::detail::vecncpy_sse<T>(src3.data(), 16, dst3.data());
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(src3[i], -dst3[i])
            << "Mismatch at index " << i << " for size " << 16;
    }

    // unaligned memory
    std::fill_n(this->unaligned_mem_vec, 31, 1.5);
    sgdlib::detail::vecncpy_sse<T>(this->unaligned_mem_vec, 32, this->unaligned_mem_vec2);
    for (size_t i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(this->unaligned_mem_vec[i], -this->unaligned_mem_vec2[i])
            << "Mismatch at index " << i << " for size " << 32;
    }

    // aligned memeory
    std::fill_n(this->aligned_mem_vec, 32, 1.0);
    sgdlib::detail::vecncpy_sse<T>(this->aligned_mem_vec, 32, this->aligned_mem_vec2);
    for (size_t i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(this->aligned_mem_vec[i], -this->aligned_mem_vec2[i])
            << "Mismatch at index " << i << " for size " << 32;
    }

    // differnent size
    std::vector<T> src4(1024, 2);
    std::vector<T> dst4(1024);

    for (size_t n : {4, 7, 15, 31, 63, 127, 255, 511, 1023}) {
        sgdlib::detail::vecncpy_sse<T>(src4.data(), n, dst4.data());
        for (size_t i = 0; i < n; ++i) {
            EXPECT_FLOAT_EQ(src4[i], -dst4[i]) ;
        }
    }

    // performance test with large size
    const std::size_t size5 = 1 << 20;
    std::vector<T> large_src1, large_dst1;
    large_src1.reserve(size5);
    large_src1.resize(size5);
    large_dst1.reserve(size5);
    large_dst1.resize(size5);
    std::fill(large_src1.begin(), large_src1.end(), 2);

    auto start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecncpy_sse<T>(large_src1.data(), size5, large_dst1.data());
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
TYPED_TEST(MathOpsSSETest, VecClipSSETest) {
    using T = typename TestFixture::Type;

    // empty vec
    sgdlib::detail::vecclip_sse<T>(nullptr, 0.0, 2.0, 10);
    sgdlib::detail::vecclip_sse<T>(this->aligned_mem_vec, 0.0, 2.0, 0);

    // Test n < 4 cases
    std::vector<T> data1 = {1.5};
    std::vector<T> expected1 = data1;
    sgdlib::detail::vecclip_ansi<T>(expected1, 0.0, 2.0);
    sgdlib::detail::vecclip_sse<T>(data1.data(), 0.0, 2.0, data1.size());
    EXPECT_EQ(data1, expected1);

    std::vector<T> data2 = {-1.0, 3.0, 0.5};
    std::vector<T> expected2 = data2;
    sgdlib::detail::vecclip_ansi<T>(expected2, 0.0, 1.0);
    sgdlib::detail::vecclip_sse<T>(data2.data(), 0.0, 1.0, data2.size());
    EXPECT_EQ(data2, expected2);

    // Test n = 4 (exactly one SIMD operation)
    std::vector<T> data3 = {-2.0, 0.5, 1.5, 3.0};
    std::vector<T> expected3 = data3;
    sgdlib::detail::vecclip_ansi<T>(expected3, 0.0, 1.0);
    sgdlib::detail::vecclip_sse<T>(data3.data(), 0.0, 1.0, data3.size());
    EXPECT_EQ(data3, expected3);

    // prime number size
    std::vector<T> data4(127, 0.0);
    for (std::size_t i = 0; i < data4.size(); ++i) {
        data4[i] = static_cast<T>(i) - 50.0;
    }

    std::vector<T> expected4 = data4;
    sgdlib::detail::vecclip_ansi<T>(expected4, -10.0, 10.0);
    sgdlib::detail::vecclip_sse<T>(data4.data(), -10.0, 10.0, data4.size());

    for (std::size_t i = 0; i < data4.size(); ++i) {
        EXPECT_FLOAT_EQ(data4[i], expected4[i]) << "Mismatch at index " << i;
    }

    // unaligned memory
    const std::size_t size3 = 1021;
    std::vector<T> unaligned_mem_vec_expected(31);
    std::fill_n(this->unaligned_mem_vec, 31, 0.0);
    for (std::size_t i = 0; i < 31; ++i) {
        this->unaligned_mem_vec[i] = static_cast<T>(i) - 50.0;
        unaligned_mem_vec_expected[i] = static_cast<T>(i) - 50.0;
    }
    sgdlib::detail::vecclip_ansi<T>(unaligned_mem_vec_expected, -10.0, 10.0);
    sgdlib::detail::vecclip_sse<T>(this->unaligned_mem_vec, -10.0, 10.0, 31);
    for (size_t i = 0; i < 31; ++i) {
        EXPECT_FLOAT_EQ(this->unaligned_mem_vec[i], unaligned_mem_vec_expected[i])
            << "Mismatch at index " << i << " for size " << 31;
    }

    // aligned memeory big size n = 1024
    const std::size_t size4 = 1024;
    std::vector<T> aligned_mem_vec_expected4(size4);
    std::fill_n(this->aligned_mem_vec, size4, 0.0);
    for (std::size_t i = 0; i < size4; ++i) {
        this->aligned_mem_vec[i] = static_cast<T>(i) - 50.0;
        aligned_mem_vec_expected4[i] = static_cast<T>(i) - 50.0;
    }
    sgdlib::detail::vecclip_ansi<T>(aligned_mem_vec_expected4, -10.0, 10.0);
    sgdlib::detail::vecclip_sse<T>(this->aligned_mem_vec, -10.0, 10.0, size4);
    for (std::size_t i = 0; i < size4; ++i) {
        EXPECT_FLOAT_EQ(this->aligned_mem_vec[i], aligned_mem_vec_expected4[i]) << "Mismatch at index " << i;
    }

    // Test NaN/inf (behavior depends on requirements)
    std::vector<T> specials = {NAN, INFINITY, -INFINITY};
    std::vector<T> spec_expected = specials;
    sgdlib::detail::vecclip_ansi<T>(spec_expected, 0.0, 1.0);
    sgdlib::detail::vecclip_sse<T>(specials.data(), 0.0, 1.0, specials.size());
    for (std::size_t i = 0; i < specials.size(); ++i) {
        EXPECT_EQ(std::isnan(specials[i]), std::isnan(spec_expected[i]));
        if (!std::isnan(specials[i])) {
            EXPECT_FLOAT_EQ(specials[i], spec_expected[i]);
        }
    }

    // large size n = 1 << 20
    const std::size_t size5 = 1 << 20;
    std::vector<T> large_vec1, large_vec2;
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
    sgdlib::detail::vecclip_sse<T>(large_vec1.data(), -10.0, 10.0, size5);
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
TYPED_TEST(MathOpsSSETest, VecHasInfSSETest) {
    using T = typename TestFixture::Type;

    T* data1 = nullptr;
    EXPECT_FALSE(sgdlib::detail::hasinf_sse<T>(data1, 0));

    // // small size array
    std::vector<T> data2 = {1.0, 2.0};
    EXPECT_FALSE(sgdlib::detail::hasinf_sse<T>(data2.data(), data2.size()));
    std::vector<T> data21 = {1.0, std::numeric_limits<T>::infinity(), 2.0};
    EXPECT_TRUE(sgdlib::detail::hasinf_sse<T>(data21.data(), data21.size()));
    std::vector<T> data22 = {1.0};
    EXPECT_FALSE(sgdlib::detail::hasinf_sse<T>(data22.data(), data22.size()));

    // aligned size array
    constexpr std::size_t size3 = 128;
    auto data3 = this->generate_test_data(size3, false);
    EXPECT_FALSE(sgdlib::detail::hasinf_sse<T>(data3.data(), size3));

    data3.back() = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(sgdlib::detail::hasinf_sse<T>(data3.data(), size3));

    // unaligned size array
    constexpr std::size_t size4 = 127;
    auto data4 = this->generate_test_data(size4, false);
    EXPECT_FALSE(sgdlib::detail::hasinf_sse<T>(data4.data(), size4));

    data4[125] = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(sgdlib::detail::hasinf_sse<T>(data4.data(), size4));

    // unaligned memory array
    std::fill_n(this->unaligned_mem_vec, size4, 0.0);
    EXPECT_FALSE(sgdlib::detail::hasinf_sse<T>(this->unaligned_mem_vec, size4));

    this->unaligned_mem_vec[125] = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(sgdlib::detail::hasinf_sse<T>(this->unaligned_mem_vec, size4));

    // aligned memory array
    std::fill_n(this->aligned_mem_vec, size4, 0.0);
    EXPECT_FALSE(sgdlib::detail::hasinf_sse<T>(this->aligned_mem_vec, size4));

    this->aligned_mem_vec[125] = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(sgdlib::detail::hasinf_sse<T>(this->aligned_mem_vec, size4));

    // large size array
    constexpr std::size_t size5 = 1'000'000;
    // without inf
    auto data5 = this->generate_test_data(size5, false);
    EXPECT_FALSE(sgdlib::detail::hasinf_sse<T>(data5.data(), size5));

    // with inf
    auto data51 = this->generate_test_data(size5, true);
    EXPECT_TRUE(sgdlib::detail::hasinf_sse<T>(data51.data(), size5));

    // edge cases
    std::vector<T> data6 = {
        std::numeric_limits<T>::min(),
        std::numeric_limits<T>::max(),
        -std::numeric_limits<T>::max()
    };
    EXPECT_FALSE(sgdlib::detail::hasinf_sse<T>(data6.data(), data6.size()));

}

// ****************************vecnorm2******************************//
TYPED_TEST(MathOpsSSETest, VecNorm2SSETest) {
    using T = typename TestFixture::Type;

    // empty vector
    std::vector<T> x1;
    EXPECT_FLOAT_EQ(0.0, sgdlib::detail::vecnorm2_sse<T>(x1.data(), x1.size(), true));
    EXPECT_FLOAT_EQ(0.0, sgdlib::detail::vecnorm2_sse<T>(x1.data(), x1.size(), false));

    // single element
    std::vector<T> x2 = {3.0};
    T expected21 = sgdlib::detail::vecnorm2_ansi<T>(x2, true);
    T expected22 = sgdlib::detail::vecnorm2_ansi<T>(x2, false);
    EXPECT_FLOAT_EQ(expected21, sgdlib::detail::vecnorm2_sse<T>(x2.data(), x2.size(), true));
    EXPECT_FLOAT_EQ(expected22, sgdlib::detail::vecnorm2_sse<T>(x2.data(), x2.size(), false));

    // multiple elements aligned
    std::vector<T> x3 = {1.0, 2.0, 3.0, 4.0}; // 4 elements for SSE alignment
    T expected31 = sgdlib::detail::vecnorm2_ansi<T>(x3, true); // 1.0 + 4.0 + 9.0 + 16.0;
    T expected32 = sgdlib::detail::vecnorm2_ansi<T>(x3, false);
    EXPECT_FLOAT_EQ(expected31, sgdlib::detail::vecnorm2_sse<T>(x3.data(), x3.size(), true));
    EXPECT_FLOAT_EQ(expected32, sgdlib::detail::vecnorm2_sse<T>(x3.data(), x3.size(), false));

    // elements with Nan
    std::vector<T> x4 = {1.0, 2.0, std::numeric_limits<T>::quiet_NaN(), 4.0};
    T expected41 = sgdlib::detail::vecnorm2_ansi(x4, true);
    T expected42 = sgdlib::detail::vecnorm2_ansi(x4, false);
    EXPECT_TRUE(std::isnan(expected41));
    EXPECT_TRUE(std::isnan(expected42));
    EXPECT_TRUE(std::isnan(sgdlib::detail::vecnorm2_sse<T>(x4.data(), x4.size(), true)));
    EXPECT_TRUE(std::isnan(sgdlib::detail::vecnorm2_sse<T>(x4.data(), x4.size(), false)));

    // multiple elements unaligned with large size
    std::size_t size5 = 1025;
    std::vector<T> x5 = this->generate_test_data(size5, false);
    T expected51 = sgdlib::detail::vecnorm2_ansi<T>(x5, true);
    T expected52 = sgdlib::detail::vecnorm2_ansi<T>(x5, false);
    EXPECT_FLOAT_EQ(expected51, sgdlib::detail::vecnorm2_sse<T>(x5.data(), x5.size(), true));
    EXPECT_FLOAT_EQ(expected52, sgdlib::detail::vecnorm2_sse<T>(x5.data(), x5.size(), false));

    // aligned memory array
    std::size_t size6 = 1024;
    std::fill_n(this->aligned_mem_vec, size6, 1.0);
    std::vector<T> x6(size6, 1.0);
    T expected61 = sgdlib::detail::vecnorm2_ansi<T>(x6, true);
    T expected62 = sgdlib::detail::vecnorm2_ansi<T>(x6, false);

    EXPECT_FLOAT_EQ(expected61, sgdlib::detail::vecnorm2_sse<T>(this->aligned_mem_vec, size6, true));
    EXPECT_FLOAT_EQ(expected62, sgdlib::detail::vecnorm2_sse<T>(this->aligned_mem_vec, size6, false));

    // overlapped memory case
    std::size_t size7 = 1000000;
    std::vector<T> x7 = this->generate_test_data(size7, false, 10.0, -10.0);
    std::vector<std::size_t> chunk_sizes = {6, 10, 25, 100, 250, 750};
    for (auto chunk_size : chunk_sizes) {
        std::size_t processed = 0;
        while (processed < size7) {
            const std::size_t current_chunk = std::min(chunk_size, size7 - processed);

            std::vector<T> chunk(
                x7.begin() + processed,
                x7.begin() + processed + current_chunk
            );

            T expected7 = sgdlib::detail::vecnorm2_ansi<T>(chunk, false);
            T actual7  = sgdlib::detail::vecnorm2_sse<T>(x7.data() + processed, current_chunk, false);
            EXPECT_NEAR(expected7, actual7, 1e-3);
            processed += current_chunk;
        }
    }

    // performance test with huge size n = 1 << 20
    std::size_t huge_size = 1 << 20;
    std::vector<T> x_huge(huge_size);
    x_huge = this->generate_test_data(huge_size, false);

    auto start = std::chrono::high_resolution_clock::now();
    T out1 = sgdlib::detail::vecnorm2_sse<T>(x_huge.data(), huge_size, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecnorm2 SIMD execution time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    T out2 = sgdlib::detail::vecnorm2_ansi<T>(x_huge, true);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "vecnorm2 ANSI execution time: " << elapsed2.count() << " seconds\n";
    std::cout << "Speedup: " << elapsed2.count() / elapsed1.count() << "x\n";

    EXPECT_FLOAT_EQ(out1, out2);

}

// ****************************vecnorm1******************************//
TYPED_TEST(MathOpsSSETest, VecNorm1SSETest) {
    using T = typename TestFixture::Type;

    // empty vector
    std::vector<T> x1;
    EXPECT_FLOAT_EQ(0.0, sgdlib::detail::vecnorm1_sse<T>(x1.data(), x1.size()));
    EXPECT_FLOAT_EQ(0.0, sgdlib::detail::vecnorm1_sse<T>(x1.data(), x1.size()));

    // small size
    // size = 1
    std::vector<T> x21 = {5.0};
    T expected21 = sgdlib::detail::vecnorm1_ansi<T>(x21);
    EXPECT_FLOAT_EQ(expected21, sgdlib::detail::vecnorm1_sse<T>(x21.data(), x21.size()));

    // size = 3
    std::vector<T> x23 = {1.0, -2.0, 3.0};
    T expected23 = sgdlib::detail::vecnorm1_ansi<T>(x23);
    EXPECT_FLOAT_EQ(expected23, sgdlib::detail::vecnorm1_sse<T>(x23.data(), x23.size()));

    // aligned size with large 4 and 16
    std::vector<T> x31 = {1.0, -2.0, 3.0, -4.0};
    T expected31 = sgdlib::detail::vecnorm1_ansi<T>(x31);
    EXPECT_FLOAT_EQ(expected31, sgdlib::detail::vecnorm1_sse<T>(x31.data(), x31.size()));

    std::size_t size32 = 16;
    std::vector<T> x32 = this->generate_test_data(size32, false);
    T expected32 = sgdlib::detail::vecnorm1_ansi<T>(x32);
    EXPECT_FLOAT_EQ(expected32, sgdlib::detail::vecnorm1_sse<T>(x32.data(), x32.size()));

    // unaligned size with large size
    std::size_t size4 = 1025;
    std::vector<T> x4 = this->generate_test_data(size4, false);
    T expected4 = sgdlib::detail::vecnorm1_ansi<T>(x4);
    EXPECT_FLOAT_EQ(expected4, sgdlib::detail::vecnorm1_sse<T>(x4.data(), x4.size()));

    // NaN cases
    std::vector<T> x5 = {
        std::numeric_limits<T>::quiet_NaN(),
        std::numeric_limits<T>::infinity(),
        -std::numeric_limits<T>::infinity(),
        1.0
    };

    T l1norm = sgdlib::detail::vecnorm1_sse(x5.data(), x5.size());
    EXPECT_TRUE(std::isnan(l1norm));

    // aligned memory array
    std::size_t size6 = 1024;
    std::fill_n(this->aligned_mem_vec, size6, 1.5);
    std::vector<T> x6(size6, 1.5);
    T expected6 = sgdlib::detail::vecnorm1_ansi<T>(x6);
    EXPECT_FLOAT_EQ(expected6, sgdlib::detail::vecnorm1_sse<T>(this->aligned_mem_vec, size6));

    // unaligned memory array
    std::size_t size7 = 1021;
    std::fill_n(this->unaligned_mem_vec, size7, 1.5);
    std::vector<T> x7(size7, 1.5);
    T expected7 = sgdlib::detail::vecnorm1_ansi<T>(x7);
    EXPECT_FLOAT_EQ(expected7, sgdlib::detail::vecnorm1_sse<T>(this->unaligned_mem_vec, size7));


    // overlapped memory case
    std::size_t size71 = 1000000;
    std::vector<T> x71 = this->generate_test_data(size71, false, 5.0, -5.0);
    std::vector<std::size_t> chunk_sizes = {6, 10, 25, 100, 250, 750};
    for (auto chunk_size : chunk_sizes) {
        std::size_t processed = 0;
        while (processed < size71) {
            const std::size_t current_chunk = std::min(chunk_size, size71 - processed);

            std::vector<T> chunk(
                x71.begin() + processed,
                x71.begin() + processed + current_chunk
            );

            T expected71 = sgdlib::detail::vecnorm1_ansi<T>(chunk);
            T actual71  = sgdlib::detail::vecnorm1_sse<T>(x71.data() + processed, current_chunk);
            EXPECT_NEAR(expected71, actual71, 1e-2);
            processed += current_chunk;
        }
    }

    // performance test with huge size n = 1 << 20
    std::size_t huge_size = 1 << 20;
    std::vector<T> x_huge(huge_size);
    x_huge = this->generate_test_data(huge_size, false);

    auto start = std::chrono::high_resolution_clock::now();
    T out1 = sgdlib::detail::vecnorm1_sse<T>(x_huge.data(), huge_size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecnorm1 SIMD execution time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    T out2 = sgdlib::detail::vecnorm1_ansi<T>(x_huge);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "vecnorm1 ANSI execution time: " << elapsed2.count() << " seconds\n";
    std::cout << "Speedup: " << elapsed2.count() / elapsed1.count() << "x\n";

    EXPECT_FLOAT_EQ(out1, out2);

}

// ****************************vecscale 1****************************//
TYPED_TEST(MathOpsSSETest, VecScaleSSETest) {
    using T = typename TestFixture::Type;

    // empty vector
    std::vector<T> out0;
    sgdlib::detail::vecscale_sse<T>(nullptr, 2.0, 10, out0.data());

    std::vector<T> empty_vec;
    sgdlib::detail::vecscale_sse<T>(empty_vec.data(), 2.0, 0, out0.data());

    std::vector<T> v3 = {1.5, 2.5, 3.5};
    std::vector<T> out3(3);
    sgdlib::detail::vecscale_sse<T>(v3.data(), 2.0, v3.size(), out3.data());
    EXPECT_FLOAT_EQ(out3[0], 3.0);
    EXPECT_FLOAT_EQ(out3[1], 5.0);
    EXPECT_FLOAT_EQ(out3[2], 7.0);

    // aligned size 4 & 8
    std::vector<T> v8 = this->generate_test_data(8, false);
    std::vector<T> out8(8);
    auto out8_copy = out8;
    sgdlib::detail::vecscale_sse<T>(v8.data(), 1.5, v8.size(), out8.data());
    sgdlib::detail::vecscale_ansi<T>(v8, 1.5, out8_copy);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(out8[i], out8_copy[i]);
    }

    std::vector<T> v7 = this->generate_test_data(7, false);
    std::vector<T> out7(7);
    auto v7_copy = v7;
    auto out7_copy = out7;
    sgdlib::detail::vecscale_sse<T>(v7.data(), 0.25, v7.size(), out7.data());
    sgdlib::detail::vecscale_ansi<T>(v7_copy, 0.25, out7_copy);
    for (size_t i = 0; i < 7; ++i) {
        EXPECT_FLOAT_EQ(out7[i], out7_copy[i]);
    }

    // large size 1025 = 1024 + 1
    std::vector<T> large(1025);
    large = this->generate_test_data(1025, false);
    std::vector<T> large_out(1025);
    auto large_copy = large;
    auto large_out_copy = large_out;
    T scale_factor1 = 1.23;
    sgdlib::detail::vecscale_sse<T>(large.data(), scale_factor1, large.size(), large_out.data());
    sgdlib::detail::vecscale_ansi<T>(large_copy, scale_factor1, large_out_copy);
    for (size_t i = 0; i < large.size(); ++i) {
        EXPECT_FLOAT_EQ(large_out[i], large_out_copy[i]);
    }

    // edage cases
    const T inf = std::numeric_limits<T>::infinity();
    const T nan = std::numeric_limits<T>::quiet_NaN();
    std::vector<T> special = {1.0, -inf, inf, nan, 0.0};
    std::vector<T> special_out(5);
    auto special_copy = special;
    auto special_out_copy = special_copy;
    T scale_factor2 = 2.0;
    sgdlib::detail::vecscale_sse<T>(special.data(), scale_factor2, special.size(), special_out.data());
    sgdlib::detail::vecscale_ansi<T>(special_copy, scale_factor2, special_out_copy);
    EXPECT_FLOAT_EQ(special_out[0], special_out_copy[0]);
    EXPECT_FLOAT_EQ(special_out[4], special_out_copy[4]);
    EXPECT_TRUE(std::isinf(special_out[1]));
    EXPECT_TRUE(std::isinf(special_out[2]));
    EXPECT_TRUE(std::isnan(special_out[3]));

    // aligned memory case
    std::size_t size6 = 1024;
    std::fill_n(this->aligned_mem_vec, size6, 1.5);
    std::vector<T> aligned_mem_vec_copy(size6, 1.5);
    std::vector<T> aligned_mem_vec_out(size6), aligned_mem_vec_copy_out(size6);

    sgdlib::detail::vecscale_sse<T>(this->aligned_mem_vec, scale_factor2, size6, aligned_mem_vec_out.data());
    sgdlib::detail::vecscale_ansi<T>(aligned_mem_vec_copy, scale_factor2, aligned_mem_vec_copy_out);
    for (size_t i = 0; i < size6; ++i) {
        EXPECT_FLOAT_EQ(aligned_mem_vec_out[i], aligned_mem_vec_copy_out[i]);
    }

    // unaligned memory case
    std::size_t size7 = 1021;
    std::fill_n(this->unaligned_mem_vec, size6, 1.5);
    std::vector<T> unaligned_mem_vec_copy(size6, 1.5);
    std::vector<T> unaligned_mem_vec_out(size6), unaligned_mem_vec_copy_out(size6);

    sgdlib::detail::vecscale_sse<T>(this->unaligned_mem_vec, scale_factor2, size6, unaligned_mem_vec_out.data());
    sgdlib::detail::vecscale_ansi<T>(unaligned_mem_vec_copy, scale_factor2, unaligned_mem_vec_copy_out);
    for (size_t i = 0; i < size7; ++i) {
        EXPECT_FLOAT_EQ(unaligned_mem_vec_out[i], unaligned_mem_vec_copy_out[i]);
    }

    // overlapped memory case
    std::size_t size71 = 1000000;
    std::vector<T> x71 = this->generate_test_data(size71, false, 10.0, -10.0);
    std::vector<std::size_t> chunk_sizes = {6, 10, 25, 100, 250, 750};
    for (auto chunk_size : chunk_sizes) {
        std::size_t processed = 0;
        while (processed < size71) {
            const std::size_t current_chunk = std::min(chunk_size, size71 - processed);
            std::vector<T> out1(current_chunk);
            std::vector<T> out2(current_chunk);
            std::vector<T> chunk(
                x71.begin() + processed,
                x71.begin() + processed + current_chunk
            );
            sgdlib::detail::vecscale_ansi<T>(chunk, scale_factor2, out1);
            sgdlib::detail::vecscale_sse<T>(x71.data() + processed, scale_factor2, current_chunk, out2.data());
            // EXPECT_NEAR(expected7, actual7, 1e-3);
            for (size_t i = 0; i < current_chunk; ++i) {
                EXPECT_FLOAT_EQ(out1[i], out2[i]);
            }
            processed += current_chunk;
        }
    }

    // performance test with huge size n = 1 << 20
    const std::size_t huge_size = 1 << 20;
    std::vector<T> x_huge(huge_size);
    std::vector<T> out_huge1(huge_size);
    std::vector<T> out_huge2(huge_size);
    x_huge = this->generate_test_data(huge_size, false);

    auto start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecscale_sse<T>(x_huge.data(), scale_factor2, huge_size, out_huge1.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecscale SIMD execution time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecscale_ansi<T>(x_huge, scale_factor2, out_huge2);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "vecscale ANSI execution time: " << elapsed2.count() << " seconds\n";
    std::cout << "Speedup: " << elapsed2.count() / elapsed1.count() << "x\n";

    for (std::size_t i = 0; i < huge_size; ++i) {
        EXPECT_FLOAT_EQ(out_huge1[i], out_huge2[i]);
    }
}

// ****************************vecscale 2****************************//

// ****************************vecadd 1******************************//
TYPED_TEST(MathOpsSSETest, VecAddWithoutConstCSSETest) {
    using T = typename TestFixture::Type;

    std::vector<T> x8;
    std::vector<T> y8;
    std::vector<T> out8;

    x8 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    y8 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    out8.resize(8);

    // empty cases
    std::size_t n8 = 8;
    sgdlib::detail::vecadd_sse<T>(nullptr, nullptr, n8, n8, nullptr);
    sgdlib::detail::vecadd_sse<T>(nullptr, y8.data(), n8, n8, out8.data());
    sgdlib::detail::vecadd_sse<T>(x8.data(), nullptr, n8, n8, out8.data());
    sgdlib::detail::vecadd_sse<T>(x8.data(), y8.data(), n8, n8, nullptr);
    sgdlib::detail::vecadd_sse<T>(x8.data(), y8.data(), n8, 7, nullptr);

    // small size n = m = 3 < 4
    std::vector<T> out3(3);
    sgdlib::detail::vecadd_sse<T>(x8.data(), y8.data(), 3, 3, out3.data());
    EXPECT_FLOAT_EQ(out3[0], x8[0] + y8[0]);
    EXPECT_FLOAT_EQ(out3[1], x8[1] + y8[1]);
    EXPECT_FLOAT_EQ(out3[2], x8[2] + y8[2]);

    // small size n = m = 5 > 4
    std::vector<T> out5(5);
    sgdlib::detail::vecadd_sse<T>(x8.data(), y8.data(), 5, 5, out5.data());
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(out5[i], x8[i] + y8[i]);
    }

    // aligned case n = m = 8
    sgdlib::detail::vecadd_sse<T>(x8.data(), y8.data(), 8, 8, out8.data());
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(out8[i], x8[i] + y8[i]);
    }

    // large size n = 1025
    std::vector<T> x_large(1025);
    std::vector<T> y_large(1025);
    std::vector<T> out1025_1(1025);
    std::vector<T> out1025_2(1025);
    x_large = this->generate_test_data(1025, false);
    y_large = this->generate_test_data(1025, false);
    sgdlib::detail::vecadd_sse<T>(x_large.data(), y_large.data(), 1025, 1025, out1025_1.data());
    sgdlib::detail::vecadd_ansi<T>(x_large, y_large, out1025_2);
    for (std::size_t i = 0; i < 1025; ++i) {
        EXPECT_FLOAT_EQ(out1025_1[i], out1025_2[i]);
    }

    // aligned memory case
    std::size_t size6 = 1024;
    std::fill_n(this->aligned_mem_vec, size6, 1.5);
    std::fill_n(this->aligned_mem_vec2, size6, 1.5);
    std::vector<T> aligned_mem_vec_copy(size6, 1.5);
    std::vector<T> aligned_mem_vec_copy2(size6, 1.5);
    std::vector<T> aligned_mem_vec_out(size6), aligned_mem_vec_copy_out(size6);

    sgdlib::detail::vecadd_sse<T>(this->aligned_mem_vec, this->aligned_mem_vec2, size6, size6, aligned_mem_vec_out.data());
    sgdlib::detail::vecadd_ansi<T>(aligned_mem_vec_copy, aligned_mem_vec_copy2, aligned_mem_vec_copy_out);
    for (size_t i = 0; i < size6; ++i) {
        EXPECT_FLOAT_EQ(aligned_mem_vec_out[i], aligned_mem_vec_copy_out[i]);
    }

    // unaligned memory case
    size6 = 1021;
    std::fill_n(this->unaligned_mem_vec, size6, 1.5);
    std::fill_n(this->unaligned_mem_vec2, size6, 1.5);
    std::vector<T> unaligned_mem_vec_copy(size6, 1.5);
    std::vector<T> unaligned_mem_vec_copy2(size6, 1.5);
    std::vector<T> unaligned_mem_vec_out(size6), unaligned_mem_vec_copy_out(size6);

    sgdlib::detail::vecadd_sse<T>(this->unaligned_mem_vec, this->unaligned_mem_vec2, size6, size6, unaligned_mem_vec_out.data());
    sgdlib::detail::vecadd_ansi<T>(unaligned_mem_vec_copy, unaligned_mem_vec_copy2, unaligned_mem_vec_copy_out);
    for (size_t i = 0; i < size6; ++i) {
        EXPECT_FLOAT_EQ(unaligned_mem_vec_out[i], unaligned_mem_vec_copy_out[i]);
    }

    // overlapped memory case
    std::size_t size71 = 1000000;
    std::vector<T> x71 = this->generate_test_data(size71, false, 10.0, -10.0);
    std::vector<std::size_t> chunk_sizes = {6, 10, 25, 100, 250, 750};
    for (auto chunk_size : chunk_sizes) {
        std::size_t processed = 0;
        while (processed < size71) {
            const std::size_t current_chunk = std::min(chunk_size, size71 - processed);
            std::vector<T> out1(current_chunk);
            std::vector<T> out2(current_chunk);
            std::vector<T> chunk(
                x71.begin() + processed,
                x71.begin() + processed + current_chunk
            );
            sgdlib::detail::vecadd_ansi<T>(chunk, chunk, out1);
            sgdlib::detail::vecadd_sse<T>(x71.data() + processed, x71.data() + processed, current_chunk, current_chunk, out2.data());
            // EXPECT_NEAR(expected7, actual7, 1e-3);
            for (size_t i = 0; i < current_chunk; ++i) {
                EXPECT_FLOAT_EQ(out1[i], out2[i]);
            }
            processed += current_chunk;
        }
    }

    // performance test with huge size n = 1 << 20
    const std::size_t huge_size = 1 << 20;
    std::vector<T> x_huge(huge_size);
    std::vector<T> y_huge(huge_size);
    std::vector<T> out_huge1(huge_size);
    std::vector<T> out_huge2(huge_size);
    x_huge = this->generate_test_data(huge_size, false);
    y_huge = this->generate_test_data(huge_size, false);

    auto start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecadd_sse<T>(x_huge.data(), y_huge.data(), huge_size, huge_size, out_huge1.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecadd without constant C SIMD execution time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecadd_ansi<T>(x_huge, y_huge, out_huge2);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "vecadd without constant C ANSI execution time: " << elapsed2.count() << " seconds\n";
    std::cout << "Speedup: " << elapsed2.count() / elapsed1.count() << "x\n";

    for (std::size_t i = 0; i < huge_size; ++i) {
        EXPECT_FLOAT_EQ(out_huge1[i], out_huge2[i]);
    }
}

// ****************************vecadd 2*******************************//
TYPED_TEST(MathOpsSSETest, VecAddWithConstCSSETest) {
    using T = typename TestFixture::Type;

    std::vector<T> x8;
    std::vector<T> y8;
    std::vector<T> out8;

    T c = 3.1415;
    x8 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    y8 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    out8.resize(8);

    // empty cases
    std::size_t n8 = 8;
    sgdlib::detail::vecadd_sse<T>(nullptr, nullptr, c, n8, n8, nullptr);
    sgdlib::detail::vecadd_sse<T>(nullptr, y8.data(), c, n8, n8, out8.data());
    sgdlib::detail::vecadd_sse<T>(x8.data(), nullptr, c, n8, n8, out8.data());
    sgdlib::detail::vecadd_sse<T>(x8.data(), y8.data(), c, n8, n8, nullptr);
    sgdlib::detail::vecadd_sse<T>(x8.data(), y8.data(), c, n8, 7, nullptr);

    // small size n = m = 3 < 4
    std::vector<T> out3(3);
    sgdlib::detail::vecadd_sse<T>(x8.data(), y8.data(), c, 3, 3, out3.data());
    EXPECT_FLOAT_EQ(out3[0], x8[0] * c + y8[0]);
    EXPECT_FLOAT_EQ(out3[1], x8[1] * c + y8[1]);
    EXPECT_FLOAT_EQ(out3[2], x8[2] * c + y8[2]);

    // small size n = m = 5 > 4
    std::vector<T> out5(5);
    sgdlib::detail::vecadd_sse<T>(x8.data(), y8.data(), c, 5, 5, out5.data());
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(out5[i], x8[i] * c + y8[i]);
    }

    // aligned case n = m = 8
    sgdlib::detail::vecadd_sse<T>(x8.data(), y8.data(), c, 8, 8, out8.data());
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(out8[i], x8[i] *c  + y8[i]);
    }

    // large size n = 1025
    std::vector<T> x_large(1025);
    std::vector<T> y_large(1025);
    std::vector<T> out1025_1(1025);
    std::vector<T> out1025_2(1025);
    x_large = this->generate_test_data(1025, false);
    y_large = this->generate_test_data(1025, false);

    sgdlib::detail::vecadd_sse<T>(x_large.data(), y_large.data(), c, 1025, 1025, out1025_1.data());
    sgdlib::detail::vecadd_ansi<T>(x_large, y_large, c, out1025_2);
    for (std::size_t i = 0; i < 1025; ++i) {
        EXPECT_FLOAT_EQ(out1025_1[i], out1025_2[i]);
    }

    // aligned memeory
    std::size_t size6 = 1024;
    std::fill_n(this->aligned_mem_vec, size6, 1.5);
    std::fill_n(this->aligned_mem_vec2, size6, 1.5);
    std::vector<T> aligned_mem_vec_copy(size6, 1.5);
    std::vector<T> aligned_mem_vec_copy2(size6, 1.5);
    std::vector<T> aligned_mem_vec_out(size6), aligned_mem_vec_copy_out(size6);

    sgdlib::detail::vecadd_sse<T>(this->aligned_mem_vec, this->aligned_mem_vec2, c, size6, size6, aligned_mem_vec_out.data());
    sgdlib::detail::vecadd_ansi<T>(aligned_mem_vec_copy, aligned_mem_vec_copy2, c, aligned_mem_vec_copy_out);
    for (size_t i = 0; i < size6; ++i) {
        EXPECT_FLOAT_EQ(aligned_mem_vec_out[i], aligned_mem_vec_copy_out[i]);
    }

    // unaligned memeory
    size6 = 1021;
    std::fill_n(this->unaligned_mem_vec, size6, 1.5);
    std::fill_n(this->unaligned_mem_vec2, size6, 1.5);
    std::vector<T> unaligned_mem_vec_copy(size6, 1.5);
    std::vector<T> unaligned_mem_vec_copy2(size6, 1.5);
    std::vector<T> unaligned_mem_vec_out(size6), unaligned_mem_vec_copy_out(size6);

    sgdlib::detail::vecadd_sse<T>(this->unaligned_mem_vec, this->unaligned_mem_vec2, c, size6, size6, unaligned_mem_vec_out.data());
    sgdlib::detail::vecadd_ansi<T>(unaligned_mem_vec_copy, unaligned_mem_vec_copy2, c, unaligned_mem_vec_copy_out);
    for (size_t i = 0; i < size6; ++i) {
        EXPECT_FLOAT_EQ(unaligned_mem_vec_out[i], unaligned_mem_vec_copy_out[i]);
    }

    // overlapped memory case
    std::size_t size71 = 1000000;
    std::vector<T> x71 = this->generate_test_data(size71, false, 10.0, -10.0);
    std::vector<std::size_t> chunk_sizes = {6, 10, 25, 100, 250, 750};
    for (auto chunk_size : chunk_sizes) {
        std::size_t processed = 0;
        while (processed < size71) {
            const std::size_t current_chunk = std::min(chunk_size, size71 - processed);
            std::vector<T> out1(current_chunk);
            std::vector<T> out2(current_chunk);
            std::vector<T> chunk(
                x71.begin() + processed,
                x71.begin() + processed + current_chunk
            );
            sgdlib::detail::vecadd_ansi<T>(chunk, chunk, c, out1);
            sgdlib::detail::vecadd_sse<T>(x71.data() + processed, x71.data() + processed, c, current_chunk, current_chunk, out2.data());
            // EXPECT_NEAR(expected7, actual7, 1e-3);
            for (size_t i = 0; i < current_chunk; ++i) {
                EXPECT_FLOAT_EQ(out1[i], out2[i]);
            }
            processed += current_chunk;
        }
    }

    // performance test with huge size n = 1 << 20
    const std::size_t huge_size = 1 << 20;
    std::vector<T> x_huge(huge_size);
    std::vector<T> y_huge(huge_size);
    std::vector<T> out_huge1(huge_size);
    std::vector<T> out_huge2(huge_size);
    x_huge = this->generate_test_data(huge_size, false);
    y_huge = this->generate_test_data(huge_size, false);

    auto start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecadd_sse<T>(x_huge.data(), y_huge.data(), c, huge_size, huge_size, out_huge1.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecadd with constant C SIMD execution time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecadd_ansi<T>(x_huge, y_huge, c, out_huge2);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "vecadd with constant C ANSI execution time: " << elapsed2.count() << " seconds\n";
    std::cout << "Speedup: " << elapsed2.count() / elapsed1.count() << "x\n";

    for (std::size_t i = 0; i < huge_size; ++i) {
        EXPECT_FLOAT_EQ(out_huge1[i], out_huge2[i]);
    }

}

// // ****************************vecdiff**********************************//
TYPED_TEST(MathOpsSSETest, VecDiffSSETest) {
    using T = typename TestFixture::Type;

    std::vector<T> x8;
    std::vector<T> y8;
    std::vector<T> out8;

    x8 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    y8 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    out8.resize(8);

    // empty cases
    std::size_t n8 = 8;
    sgdlib::detail::vecdiff_sse<T>(nullptr, nullptr, n8, n8, nullptr);
    sgdlib::detail::vecdiff_sse<T>(nullptr, y8.data(), n8, n8, out8.data());
    sgdlib::detail::vecdiff_sse<T>(x8.data(), nullptr, n8, n8, out8.data());
    sgdlib::detail::vecdiff_sse<T>(x8.data(), y8.data(), n8, n8, nullptr);
    sgdlib::detail::vecdiff_sse<T>(x8.data(), y8.data(), n8, 7, nullptr);

    // small size n = m = 3 < 4
    std::vector<T> out3(3);
    sgdlib::detail::vecdiff_sse<T>(x8.data(), y8.data(), 3, 3, out3.data());
    EXPECT_FLOAT_EQ(out3[0], x8[0] - y8[0]);
    EXPECT_FLOAT_EQ(out3[1], x8[1] - y8[1]);
    EXPECT_FLOAT_EQ(out3[2], x8[2] - y8[2]);

    // small size n = m = 5 > 4
    std::vector<T> out5(5);
    sgdlib::detail::vecdiff_sse<T>(x8.data(), y8.data(), 5, 5, out5.data());
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(out5[i], x8[i] - y8[i]);
    }

    // aligned case n = m = 8
    sgdlib::detail::vecdiff_sse<T>(x8.data(), y8.data(), 8, 8, out8.data());
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(out8[i], x8[i] - y8[i]);
    }

    // large size n = 1025
    std::vector<T> x_large(1025);
    std::vector<T> y_large(1025);
    std::vector<T> out1025_1(1025);
    std::vector<T> out1025_2(1025);
    x_large = this->generate_test_data(1025, false);
    y_large = this->generate_test_data(1025, false);

    sgdlib::detail::vecdiff_sse<T>(x_large.data(), y_large.data(), 1025, 1025, out1025_1.data());
    sgdlib::detail::vecdiff_ansi<T>(x_large, y_large, out1025_2);
    for (std::size_t i = 0; i < 1025; ++i) {
        EXPECT_FLOAT_EQ(out1025_1[i], out1025_2[i]);
    }

    // aligned memeory
    std::size_t size6 = 1024;
    std::fill_n(this->aligned_mem_vec, size6, 2.5);
    std::fill_n(this->aligned_mem_vec2, size6, 1.5);
    std::vector<T> aligned_mem_vec_copy(size6, 2.5);
    std::vector<T> aligned_mem_vec_copy2(size6, 1.5);
    std::vector<T> aligned_mem_vec_out(size6), aligned_mem_vec_copy_out(size6);

    sgdlib::detail::vecdiff_sse<T>(this->aligned_mem_vec, this->aligned_mem_vec2, size6, size6, aligned_mem_vec_out.data());
    sgdlib::detail::vecdiff_ansi<T>(aligned_mem_vec_copy, aligned_mem_vec_copy2, aligned_mem_vec_copy_out);
    for (size_t i = 0; i < size6; ++i) {
        EXPECT_FLOAT_EQ(aligned_mem_vec_out[i], aligned_mem_vec_copy_out[i]);
    }

    // unaligned memeory
    size6 = 1021;
    std::fill_n(this->unaligned_mem_vec, size6, 2.5);
    std::fill_n(this->unaligned_mem_vec2, size6, 1.5);
    std::vector<T> unaligned_mem_vec_copy(size6, 2.5);
    std::vector<T> unaligned_mem_vec_copy2(size6, 1.5);
    std::vector<T> unaligned_mem_vec_out(size6), unaligned_mem_vec_copy_out(size6);

    sgdlib::detail::vecdiff_sse<T>(this->unaligned_mem_vec, this->unaligned_mem_vec2, size6, size6, unaligned_mem_vec_out.data());
    sgdlib::detail::vecdiff_ansi<T>(unaligned_mem_vec_copy, unaligned_mem_vec_copy2, unaligned_mem_vec_copy_out);
    for (size_t i = 0; i < size6; ++i) {
        EXPECT_FLOAT_EQ(unaligned_mem_vec_out[i], unaligned_mem_vec_copy_out[i]);
    }

    // overlapped memory case
    std::size_t size71 = 1000000;
    std::vector<T> x71 = this->generate_test_data(size71, false, 10.0, -10.0);
    std::vector<T> x72 = this->generate_test_data(size71, false, 10.0, -10.0);
    std::vector<std::size_t> chunk_sizes = {6, 10, 25, 100, 250, 750};
    for (auto chunk_size : chunk_sizes) {
        std::size_t processed = 0;
        while (processed < size71) {
            const std::size_t current_chunk = std::min(chunk_size, size71 - processed);
            std::vector<T> out1(current_chunk);
            std::vector<T> out2(current_chunk);
            std::vector<T> chunk1(
                x71.begin() + processed,
                x71.begin() + processed + current_chunk
            );
            std::vector<T> chunk2(
                x72.begin() + processed,
                x72.begin() + processed + current_chunk
            );
            sgdlib::detail::vecdiff_ansi<T>(chunk1, chunk2, out1);
            sgdlib::detail::vecdiff_sse<T>(x71.data() + processed, x72.data() + processed, current_chunk, current_chunk, out2.data());
            // EXPECT_NEAR(expected7, actual7, 1e-3);
            for (size_t i = 0; i < current_chunk; ++i) {
                EXPECT_FLOAT_EQ(out1[i], out2[i]);
            }
            processed += current_chunk;
        }
    }

    // performance test with huge size n = 1 << 20
    const std::size_t huge_size = 1 << 20;
    std::vector<T> x_huge(huge_size);
    std::vector<T> y_huge(huge_size);
    std::vector<T> out_huge1(huge_size);
    std::vector<T> out_huge2(huge_size);
    x_huge = this->generate_test_data(huge_size, false);
    y_huge = this->generate_test_data(huge_size, false);

    auto start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecdiff_sse<T>(x_huge.data(), y_huge.data(), huge_size, huge_size, out_huge1.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecdiff SIMD execution time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecdiff_ansi<T>(x_huge, y_huge, out_huge2);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "vecdiff ANSI execution time: " << elapsed2.count() << " seconds\n";
    std::cout << "Speedup: " << elapsed2.count() / elapsed1.count() << "x\n";

    for (std::size_t i = 0; i < huge_size; ++i) {
        EXPECT_FLOAT_EQ(out_huge1[i], out_huge2[i]);
    }
}

// // ****************************vecdot**********************************//
TYPED_TEST(MathOpsSSETest, VecDotSSETest) {
    using T = typename TestFixture::Type;

    // empty x / y
    std::vector<T> x1 = {1.0, 2.0};
    EXPECT_FLOAT_EQ(sgdlib::detail::vecdot_sse<T>(nullptr, x1.data(), 2, 2), 0.0);
    EXPECT_FLOAT_EQ(sgdlib::detail::vecdot_sse<T>(x1.data(), nullptr, 2, 2), 0.0);

    // x length != y length
    std::vector<T> x2 = {1.0, 2.0};
    std::vector<T> y2 = {1.0, 2.0, 3.0};
    EXPECT_FLOAT_EQ(sgdlib::detail::vecdot_sse<T>(x2.data(), y2.data(), 2, 3), 0.0);

    // small size
    std::vector<T> x3 = {1.0, 2.0, 3.0};  // n=3
    std::vector<T> y3 = {4.0, 5.0, 6.0};
    // 1.0*4.0 + 2.0*5.0 + 3.0*6.0;
    T expected3 = sgdlib::detail::vecdot_ansi<T>(x3, y3);
    EXPECT_FLOAT_EQ(sgdlib::detail::vecdot_sse<T>(x3.data(), y3.data(), 3, 3), expected3);

    // unaligned small size
    std::vector<T> x4 = {1.0, 2.0, 3.0, 4.0, 5.0};  // n=5
    std::vector<T> y4 = {2.0, 3.0, 4.0, 5.0, 6.0};
    T expected4 = sgdlib::detail::vecdot_ansi<T>(x4, y4);;
    EXPECT_FLOAT_EQ(sgdlib::detail::vecdot_sse<T>(x4.data(), y4.data(), 5, 5), expected4);

    // unaligned big size 1024
    std::size_t size1 = 1025;
    std::vector<T> x5(size1), y5(size1);
    x5 = this->generate_test_data(size1, false, 1, -1);
    y5 = this->generate_test_data(size1, false, 1, -1);

    T expected5 = sgdlib::detail::vecdot_ansi<T>(x5, y5);
    T result5 = sgdlib::detail::vecdot_sse<T>(x5.data(), y5.data(), size1, size1);
    EXPECT_NEAR(result5, expected5, 1e-4);

    // aligned memeory
    std::size_t size6 = 1024;
    std::fill_n(this->aligned_mem_vec, size6, 2.5);
    std::fill_n(this->aligned_mem_vec2, size6, 1.);
    std::vector<T> aligned_mem_vec_copy(size6, 2.5);
    std::vector<T> aligned_mem_vec_copy2(size6, 1.);

    T result61 = sgdlib::detail::vecdot_sse<T>(this->aligned_mem_vec, this->aligned_mem_vec2, size6, size6);
    T expected61 = sgdlib::detail::vecdot_ansi<T>(aligned_mem_vec_copy, aligned_mem_vec_copy2);
    EXPECT_NEAR(result61, expected61, 1e-4);

    // aligned memeory
    size6 = 1021;
    std::fill_n(this->unaligned_mem_vec, size6, 2.5);
    std::fill_n(this->unaligned_mem_vec2, size6, 1.);
    std::vector<T> unaligned_mem_vec_copy(size6, 2.5);
    std::vector<T> unaligned_mem_vec_copy2(size6, 1.);

    T result62 = sgdlib::detail::vecdot_sse<T>(this->unaligned_mem_vec, this->unaligned_mem_vec2, size6, size6);
    T expected62 = sgdlib::detail::vecdot_ansi<T>(unaligned_mem_vec_copy, unaligned_mem_vec_copy2);
    EXPECT_NEAR(result62, expected62, 1e-4);

    // overlapped memory case
    std::size_t size7 = 1000000;
    std::vector<T> x71 = this->generate_test_data(size7, false, 1.0, -1.0);
    std::vector<T> x72 = this->generate_test_data(size7, false, 1.0, -1.0);
    std::vector<std::size_t> chunk_sizes = {6, 10, 25, 100, 250, 750};
    for (auto chunk_size : chunk_sizes) {
        std::size_t processed = 0;
        while (processed < size7) {
            const std::size_t current_chunk = std::min(chunk_size, size7 - processed);

            std::vector<T> chunk1(
                x71.begin() + processed,
                x71.begin() + processed + current_chunk
            );

            std::vector<T> chunk2(
                x72.begin() + processed,
                x72.begin() + processed + current_chunk
            );

            T expected7 = sgdlib::detail::vecdot_ansi<T>(chunk1, chunk2);
            T actual7  = sgdlib::detail::vecdot_sse<T>(x71.data() + processed, x72.data() + processed, current_chunk, current_chunk);
            EXPECT_NEAR(expected7, actual7, 1e-3);
            processed += current_chunk;
        }
    }

    // performance test
    const std::size_t size2 = 1 << 20;  // 1M elements
    std::vector<T> x6(size2), y6(size2);
    x6 = this->generate_test_data(size2, false, 1, -1);
    y6 = this->generate_test_data(size2, false, 1, -1);

    auto start = std::chrono::high_resolution_clock::now();
    T result6 = sgdlib::detail::vecdot_sse<T>(x6.data(), y6.data(), size2, size2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecdot SIMD execution time: " << elapsed1.count() << " s\n";

    start = std::chrono::high_resolution_clock::now();
    T expected6 = sgdlib::detail::vecdot_ansi<T>(x6, y6);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "vecdot ANSI execution time: " << elapsed2.count() << " s\n";
    std::cout << "Speedup: " << elapsed2.count() / elapsed1.count() << "x\n";

    EXPECT_NEAR(result6, expected6, 1e-2);
}

// // ****************************vecmul**********************************//
TYPED_TEST(MathOpsSSETest, VecMulSSETest) {
    using T = typename TestFixture::Type;

    // empty x / y and x length!= y length
    std::vector<T> small_input1(3), small_input2(3), small_output(3);
    sgdlib::detail::vecmul_sse<T>(small_input1.data(), nullptr, 3, 3, small_output.data());
    sgdlib::detail::vecmul_sse<T>(nullptr, small_input2.data(), 3, 3, small_output.data());
    sgdlib::detail::vecmul_sse<T>(small_input1.data(), small_input2.data(), 3, 3, nullptr);
    sgdlib::detail::vecmul_sse<T>(small_input1.data(), small_input2.data(), 3, 4, small_output.data());

    // small size n = 3
    small_input1 = {1.0, 2.0, 3.0};
    small_input2 = {4.0, 5.0, 6.0};
    std::vector<T> small_expected = {4.0, 10.0, 18.0};
    sgdlib::detail::vecmul_sse<T>(small_input1.data(), small_input2.data(), 3, 3, small_output.data());

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(small_output[i], small_expected[i]) << "Mismatch at index " << i << " for size " << 3;
    }

    // n = 7
    std::vector<T> input1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    std::vector<T> input2 = {6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    std::vector<T> expected1(7); // {6.0, 14.0, 24.0, 36.0, 50.0, 66.0, 84.0};
    std::vector<T> out(7);

    sgdlib::detail::vecmul_ansi<T>(input1, input2, expected1);
    sgdlib::detail::vecmul_sse<T>(input1.data(), input2.data(), 7, 7, out.data());

    for (std::size_t i = 0; i < 7; ++i) {
        EXPECT_FLOAT_EQ(expected1[i], out[i]) << "Mismatch at index " << i << " for size " << 3;
    }

    // middle size
    const std::size_t n = 1025;
    std::vector<T> x6(n), y6(n);
    std::vector<T> out6(n);
    std::vector<T> expected6(n);
    x6 = this->generate_test_data(n, false, 1, -1);
    y6 = this->generate_test_data(n, false, 1, -1);

    sgdlib::detail::vecmul_ansi<T>(x6, y6, expected6);
    sgdlib::detail::vecmul_sse<T>(x6.data(), y6.data(), n, n, out6.data());

    for (std::size_t i = 0; i < 7; ++i) {
        EXPECT_FLOAT_EQ(expected6[i], out6[i]) << "Mismatch at index " << i << " for size " << 3;
    }

    // aligned memeory
    std::size_t size6 = 1024;
    std::fill_n(this->aligned_mem_vec, size6, 2.5);
    std::fill_n(this->aligned_mem_vec2, size6, 1.5);
    std::vector<T> aligned_mem_vec_copy(size6, 2.5);
    std::vector<T> aligned_mem_vec_copy2(size6, 1.5);
    std::vector<T> aligned_mem_vec_out(size6), aligned_mem_vec_copy_out(size6);

    sgdlib::detail::vecmul_sse<T>(this->aligned_mem_vec, this->aligned_mem_vec2, size6, size6, aligned_mem_vec_out.data());
    sgdlib::detail::vecmul_ansi<T>(aligned_mem_vec_copy, aligned_mem_vec_copy2, aligned_mem_vec_copy_out);
    for (size_t i = 0; i < size6; ++i) {
        EXPECT_FLOAT_EQ(aligned_mem_vec_out[i], aligned_mem_vec_copy_out[i]);
    }

    // unaligned memeory
    size6 = 1024;
    std::fill_n(this->unaligned_mem_vec, size6, 2.5);
    std::fill_n(this->unaligned_mem_vec2, size6, 1.5);
    std::vector<T> unaligned_mem_vec_copy(size6, 2.5);
    std::vector<T> unaligned_mem_vec_copy2(size6, 1.5);
    std::vector<T> unaligned_mem_vec_out(size6), unaligned_mem_vec_copy_out(size6);

    sgdlib::detail::vecmul_sse<T>(this->unaligned_mem_vec, this->unaligned_mem_vec2, size6, size6, unaligned_mem_vec_out.data());
    sgdlib::detail::vecmul_ansi<T>(unaligned_mem_vec_copy, unaligned_mem_vec_copy2, unaligned_mem_vec_copy_out);
    for (size_t i = 0; i < size6; ++i) {
        EXPECT_FLOAT_EQ(unaligned_mem_vec_out[i], unaligned_mem_vec_copy_out[i]);
    }

    // specials float number
    std::vector<T> input3 = {0.0f, -0.0f, INFINITY, -INFINITY, NAN, 1.0f};
    std::vector<T> input31 = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, NAN};
    std::vector<T> expected3(6);
    std::vector<T> out3(6);

    sgdlib::detail::vecmul_sse<T>(input3.data(), input31.data(), 6, 6, out3.data());
    sgdlib::detail::vecmul_ansi<T>(input3, input31, expected3);
    for (size_t i = 0; i < 6; ++i) {
        if (!std::isnan(expected3[i])) {
            EXPECT_FLOAT_EQ(out3[i], expected3[i]);
        } else {
            EXPECT_TRUE(std::isnan(out3[i]));
        }
    }

    // overlapped memory case
    std::size_t size7 = 1000000;
    std::vector<T> x71 = this->generate_test_data(size7, false, 1.0, -1.0);
    std::vector<T> x72 = this->generate_test_data(size7, false, 1.0, -1.0);
    std::vector<std::size_t> chunk_sizes = {6, 10, 25, 100, 250, 750};
    for (auto chunk_size : chunk_sizes) {
        std::size_t processed = 0;
        while (processed < size7) {
            const std::size_t current_chunk = std::min(chunk_size, size7 - processed);
            std::vector<T> out1(current_chunk);
            std::vector<T> out2(current_chunk);
            std::vector<T> chunk1(
                x71.begin() + processed,
                x71.begin() + processed + current_chunk
            );

            std::vector<T> chunk2(
                x72.begin() + processed,
                x72.begin() + processed + current_chunk
            );

            sgdlib::detail::vecmul_ansi<T>(chunk1, chunk2, out1);
            sgdlib::detail::vecmul_sse<T>(x71.data() + processed, x72.data() + processed, current_chunk, current_chunk, out2.data());
            for (size_t i = 0; i < current_chunk; ++i) {
                EXPECT_FLOAT_EQ(out1[i], out2[i]);
            }
            processed += current_chunk;
        }
    }

    // performance test
    const std::size_t size2 = 1 << 20;  // 1M elements
    std::vector<T> x(size2), y(size2), out1(size2), out2(size2);
    x = this->generate_test_data(size2, false, 1, -1);
    y = this->generate_test_data(size2, false, 1, -1);

    auto start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecmul_sse<T>(x.data(), y.data(), size2, size2, out1.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecdot SIMD execution time: " << elapsed1.count() << " s\n";

    start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecmul_ansi<T>(x, y, out2);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "vecdot ANSI execution time: " << elapsed2.count() << " s\n";
    std::cout << "Speedup: " << elapsed2.count() / elapsed1.count() << "x\n";

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(out1[i], out2[i]);
    }
}
