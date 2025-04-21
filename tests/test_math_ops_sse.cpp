#include <gtest/gtest.h>
#include <chrono>

#include "sgdlib/common/prereqs.hpp"
#include "sgdlib/math/kernels/ops_ansi.hpp"
#include "sgdlib/math/kernels/ops_sse_double.hpp"
#include "sgdlib/math/kernels/ops_sse_float.hpp"

// ****************************generate testing data*******************************//
template<typename T>
class MathOpsSSETest : public ::testing::Test {
protected:
    using Type = T;

    void SetUp() override {
        engine_.seed(42);
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
};

using TestValueTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MathOpsSSETest, TestValueTypes);

// ****************************vecclip*******************************//
template <typename T>
void vecclip_sse_wrapper(T min, T max, T* x, std::size_t n) {
    if constexpr (std::is_same_v<T, float>) {
        sgdlib::detail::vecclip_sse_float(x, min, max, n);
    }
    else if constexpr (std::is_same_v<T, double>) {
        sgdlib::detail::vecclip_sse_double(x, min, max, n);
    }
}

TYPED_TEST(MathOpsSSETest, VecClipSSETest) {
    using T = typename TestFixture::Type;

    // Test n < 4 cases
    std::vector<T> data1 = {1.5};
    std::vector<T> expected1 = data1;
    sgdlib::detail::vecclip_ansi<T>(expected1, 0.0, 2.0);
    vecclip_sse_wrapper<T>(0.0, 2.0, data1.data(), data1.size());
    EXPECT_EQ(data1, expected1);

    std::vector<T> data2 = {-1.0, 3.0, 0.5};
    std::vector<T> expected2 = data2;
    sgdlib::detail::vecclip_ansi<T>(expected2, 0.0, 1.0);
    vecclip_sse_wrapper<T>(0.0, 1.0, data2.data(), data2.size());
    EXPECT_EQ(data2, expected2);

    // Test n = 4 (exactly one SIMD operation)
    std::vector<T> data3 = {-2.0, 0.5, 1.5, 3.0};
    std::vector<T> expected3 = data3;
    sgdlib::detail::vecclip_ansi<T>(expected3, 0.0, 1.0);
    vecclip_sse_wrapper<T>(0.0, 1.0, data3.data(), data3.size());
    EXPECT_EQ(data3, expected3);

    // Test n > 4 with remainder
    std::vector<T> data4(127, 0.0); // Prime number size
    for (std::size_t i = 0; i < data4.size(); ++i) {
        data4[i] = static_cast<T>(i) - 50.0;
    }

    std::vector<T> expected4 = data4;
    sgdlib::detail::vecclip_ansi<T>(expected4, -10.0, 10.0);
    vecclip_sse_wrapper<T>(-10.0, 10.0, data4.data(), data4.size());

    for (std::size_t i = 0; i < data4.size(); ++i) {
        EXPECT_FLOAT_EQ(data4[i], expected4[i]) << "Mismatch at index " << i;
    }

    // Test NaN/inf (behavior depends on requirements)
    std::vector<T> specials = {NAN, INFINITY, -INFINITY};
    std::vector<T> spec_expected = specials;
    sgdlib::detail::vecclip_ansi<T>(spec_expected, 0.0, 1.0);
    vecclip_sse_wrapper<T>(0.0, 1.0, specials.data(), specials.size());
    for (std::size_t i = 0; i < specials.size(); ++i) {
        EXPECT_EQ(std::isnan(specials[i]), std::isnan(spec_expected[i]));
        if (!std::isnan(specials[i])) {
            EXPECT_FLOAT_EQ(specials[i], spec_expected[i]);
        }
    }
}

// *****************************hasinf*******************************//
template <typename T>
bool hasinf_sse_wrapper(const T* x, std::size_t n) {
    if constexpr (std::is_same_v<T, float>) {
        return sgdlib::detail::hasinf_sse_float(x, n);
    }
    else if constexpr (std::is_same_v<T, double>) {
        return sgdlib::detail::hasinf_sse_double(x, n);
    }
    return false;
}

TYPED_TEST(MathOpsSSETest, VecHasInfSSETest) {
    using T = typename TestFixture::Type;

    T* data1 = nullptr;
    EXPECT_FALSE(hasinf_sse_wrapper<T>(data1, 0));

    // // small size array
    std::vector<T> data2 = {1.0, 2.0};
    EXPECT_FALSE(hasinf_sse_wrapper<T>(data2.data(), data2.size()));
    std::vector<T> data21 = {1.0, std::numeric_limits<T>::infinity()};
    EXPECT_TRUE(hasinf_sse_wrapper<T>(data21.data(), data21.size()));
    std::vector<T> data22 = {1.0};
    EXPECT_FALSE(hasinf_sse_wrapper<T>(data22.data(), data22.size()));

    // aligned size array
    constexpr std::size_t size3 = 128;
    auto data3 = this->generate_test_data(size3, false);
    EXPECT_FALSE(hasinf_sse_wrapper<T>(data3.data(), size3));

    data3.back() = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(hasinf_sse_wrapper<T>(data3.data(), size3));

    // unaligned size array
    constexpr std::size_t size4 = 127;
    auto data4 = this->generate_test_data(size4, false);
    EXPECT_FALSE(hasinf_sse_wrapper<T>(data4.data(), size4));

    data4[125] = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(hasinf_sse_wrapper<T>(data4.data(), size4));

    // large size array
    constexpr std::size_t size5 = 1'000'000;
    // without inf
    auto data5 = this->generate_test_data(size5, false);
    EXPECT_FALSE(hasinf_sse_wrapper<T>(data5.data(), size5));

    // with inf
    auto data51 = this->generate_test_data(size5, true);
    EXPECT_TRUE(hasinf_sse_wrapper<T>(data51.data(), size5));

    // edge cases
    std::vector<T> data6 = {
        std::numeric_limits<T>::min(),
        std::numeric_limits<T>::max(),
        -std::numeric_limits<T>::max()
    };
    EXPECT_FALSE(hasinf_sse_wrapper<T>(data6.data(), data6.size()));
}

// ****************************vecnorm2******************************//
template <typename T>
T vecnorm2_sse_wrapper(const T* x, std::size_t n, bool squared) {
    if constexpr (std::is_same_v<T, float>) {
        return sgdlib::detail::vecnorm2_sse_float(x, n, squared);
    }
    else if constexpr (std::is_same_v<T, double>) {
        return sgdlib::detail::vecnorm2_sse_double(x, n, squared);
    }
    return 0.0;
}

TYPED_TEST(MathOpsSSETest, VecNorm2SSETest) {
    using T = typename TestFixture::Type;

    // empty vector
    std::vector<T> x1;
    EXPECT_FLOAT_EQ(0.0, vecnorm2_sse_wrapper<T>(x1.data(), x1.size(), true));
    EXPECT_FLOAT_EQ(0.0, vecnorm2_sse_wrapper<T>(x1.data(), x1.size(), false));

    // single element
    std::vector<T> x2 = {3.0};
    T expected21 = sgdlib::detail::vecnorm2_ansi<T>(x2, true);
    T expected22 = sgdlib::detail::vecnorm2_ansi<T>(x2, false);
    EXPECT_FLOAT_EQ(expected21, vecnorm2_sse_wrapper<T>(x2.data(), x2.size(), true));
    EXPECT_FLOAT_EQ(expected22, vecnorm2_sse_wrapper<T>(x2.data(), x2.size(), false));

    // multiple elements aligned
    std::vector<T> x3 = {1.0, 2.0, 3.0, 4.0}; // 4 elements for SSE alignment
    T expected31 = sgdlib::detail::vecnorm2_ansi<T>(x3, true); // 1.0 + 4.0 + 9.0 + 16.0;
    T expected32 = sgdlib::detail::vecnorm2_ansi<T>(x3, false);
    EXPECT_FLOAT_EQ(expected31, vecnorm2_sse_wrapper<T>(x3.data(), x3.size(), true));
    EXPECT_FLOAT_EQ(expected32, vecnorm2_sse_wrapper<T>(x3.data(), x3.size(), false));

    // elements with Nan
    std::vector<T> x4 = {1.0, 2.0, std::numeric_limits<T>::quiet_NaN(), 4.0};
    T expected41 = sgdlib::detail::vecnorm2_ansi(x4, true);
    T expected42 = sgdlib::detail::vecnorm2_ansi(x4, false);
    EXPECT_TRUE(std::isnan(expected41));
    EXPECT_TRUE(std::isnan(expected42));
    EXPECT_TRUE(std::isnan(vecnorm2_sse_wrapper<T>(x4.data(), x4.size(), true)));
    EXPECT_TRUE(std::isnan(vecnorm2_sse_wrapper<T>(x4.data(), x4.size(), false)));

    // multiple elements unaligned with large size
    std::size_t size5 = 1025;
    std::vector<T> x5 = this->generate_test_data(size5, false);
    T expected51 = sgdlib::detail::vecnorm2_ansi<T>(x5, true);
    T expected52 = sgdlib::detail::vecnorm2_ansi<T>(x5, false);
    EXPECT_FLOAT_EQ(expected51, vecnorm2_sse_wrapper<T>(x5.data(), x5.size(), true));
    EXPECT_FLOAT_EQ(expected52, vecnorm2_sse_wrapper<T>(x5.data(), x5.size(), false));

}

// ****************************vecnorm1******************************//
template <typename T>
T vecnorm1_sse_wrapper(const T* x, std::size_t n) {
    if constexpr (std::is_same_v<T, float>) {
        return sgdlib::detail::vecnorm1_sse_float(x, n);
    }
    else if constexpr (std::is_same_v<T, double>) {
        return sgdlib::detail::vecnorm1_sse_double(x, n);
    }
    return 0.0;
}

TYPED_TEST(MathOpsSSETest, VecNorm1SSETest) {
    using T = typename TestFixture::Type;

    // empty vector
    std::vector<T> x1;
    EXPECT_FLOAT_EQ(0.0, vecnorm1_sse_wrapper<T>(x1.data(), x1.size()));
    EXPECT_FLOAT_EQ(0.0, vecnorm1_sse_wrapper<T>(x1.data(), x1.size()));

    // small size
    // size = 1
    std::vector<T> x21 = {5.0};
    T expected21 = sgdlib::detail::vecnorm1_ansi<T>(x21);
    EXPECT_FLOAT_EQ(expected21, vecnorm1_sse_wrapper<T>(x21.data(), x21.size()));

    // size = 2
    std::vector<T> x22 = {3.0, -4.0};
    T expected22 = sgdlib::detail::vecnorm1_ansi<T>(x22);
    EXPECT_FLOAT_EQ(expected22, vecnorm1_sse_wrapper<T>(x22.data(), x22.size()));

    // size = 3
    std::vector<T> x23 = {1.0, -2.0, 3.0};
    T expected23 = sgdlib::detail::vecnorm1_ansi<T>(x23);
    EXPECT_FLOAT_EQ(expected23, vecnorm1_sse_wrapper<T>(x23.data(), x23.size()));

    // aligned size with large 4 and 16
    std::vector<T> x31 = {1.0, -2.0, 3.0, -4.0};
    T expected31 = sgdlib::detail::vecnorm1_ansi<T>(x31);
    EXPECT_FLOAT_EQ(expected31, vecnorm1_sse_wrapper<T>(x31.data(), x31.size()));

    std::size_t size32 = 16;
    std::vector<T> x32 = this->generate_test_data(size32, false);
    T expected32 = sgdlib::detail::vecnorm1_ansi<T>(x32);
    EXPECT_FLOAT_EQ(expected32, vecnorm1_sse_wrapper<T>(x32.data(), x32.size()));

    // unaligned size with large size
    std::size_t size4 = 1025;
    std::vector<T> x4 = this->generate_test_data(size4, false);
    T expected4 = sgdlib::detail::vecnorm1_ansi<T>(x4);
    EXPECT_FLOAT_EQ(expected4, vecnorm1_sse_wrapper<T>(x4.data(), x4.size()));

    // NaN cases
    std::vector<T> x5 = {
        std::numeric_limits<T>::quiet_NaN(),
        std::numeric_limits<T>::infinity(),
        -std::numeric_limits<T>::infinity(),
        1.0
    };

    T l1norm = vecnorm1_sse_wrapper(x5.data(), x5.size());
    EXPECT_TRUE(std::isnan(l1norm));
}

// ****************************vecscale 1****************************//
template <typename T>
void vecscale_sse_wrapper(const T* x, const T c, std::size_t n, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        sgdlib::detail::vecscale_sse_float(x, c, n, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        sgdlib::detail::vecscale_sse_double(x, c, n, out);
    }
}

TYPED_TEST(MathOpsSSETest, VecScaleSSETest) {
    using T = typename TestFixture::Type;

    // empty vector
    std::vector<T> out0;
    vecscale_sse_wrapper<T>(nullptr, 2.0, 10, out0.data());

    std::vector<T> empty_vec;
    vecscale_sse_wrapper<T>(empty_vec.data(), 2.0, 0, out0.data());

    // small size
    std::vector<T> v1 = {3.0};
    std::vector<T> out1(1);
    vecscale_sse_wrapper<T>(v1.data(), 2.0, v1.size(), out1.data());
    EXPECT_FLOAT_EQ(out1[0], 6.0);

    std::vector<T> v2 = {1.0, -2.0};
    std::vector<T> out2(2);
    vecscale_sse_wrapper<T>(v2.data(), 3.0, v2.size(), out2.data());
    EXPECT_FLOAT_EQ(out2[0], 3.0);
    EXPECT_FLOAT_EQ(out2[1], -6.0);

    std::vector<T> v3 = {1.5, 2.5, 3.5};
    std::vector<T> out3(3);
    vecscale_sse_wrapper<T>(v3.data(), 2.0, v3.size(), out3.data());
    EXPECT_FLOAT_EQ(out3[0], 3.0);
    EXPECT_FLOAT_EQ(out3[1], 5.0);
    EXPECT_FLOAT_EQ(out3[2], 7.0);

    // aligned size 4 & 8
    std::vector<T> v4 = {1.0, 2.0, 3.0, 4.0};
    std::vector<T> out4(4);
    // auto v4_copy = v4;
    auto out4_copy = out4;
    vecscale_sse_wrapper<T>(v4.data(), 0.5, v4.size(), out4.data());
    sgdlib::detail::vecscale_ansi<T>(v4, 0.5, out4_copy);
    EXPECT_EQ(out4, out4_copy);

    std::vector<T> v8 = this->generate_test_data(8, false);
    std::vector<T> out8(8);
    auto out8_copy = out8;
    vecscale_sse_wrapper<T>(v8.data(), 1.5, v8.size(), out8.data());
    sgdlib::detail::vecscale_ansi<T>(v8, 1.5, out8_copy);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(out8[i], out8_copy[i]);
    }

    // unaligned size
    std::vector<T> v5 = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<T> out5(5);
    std::vector<T> out5_copy(5);
    vecscale_sse_wrapper<T>(v5.data(), 2.0, v5.size(), out5.data());
    sgdlib::detail::vecscale_ansi<T>(v5, 2.0, out5_copy);
    // EXPECT_EQ(out5, out5_copy);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(out5[i], out5_copy[i]);
    }

    std::vector<T> v7 = this->generate_test_data(7, false);
    std::vector<T> out7(7);
    auto v7_copy = v7;
    auto out7_copy = out7;
    vecscale_sse_wrapper<T>(v7.data(), 0.25, v7.size(), out7.data());
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

    vecscale_sse_wrapper<T>(large.data(), scale_factor1, large.size(), large_out.data());
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

    vecscale_sse_wrapper<T>(special.data(), scale_factor2, special.size(), special_out.data());
    sgdlib::detail::vecscale_ansi<T>(special_copy, scale_factor2, special_out_copy);

    EXPECT_FLOAT_EQ(special_out[0], special_out_copy[0]);
    EXPECT_FLOAT_EQ(special_out[4], special_out_copy[4]);

    EXPECT_TRUE(std::isinf(special_out[1]));
    EXPECT_TRUE(std::isinf(special_out[2]));

    EXPECT_TRUE(std::isnan(special_out[3]));

}

// ****************************vecscale 2****************************//



// ****************************vecadd 1******************************//
template <typename T>
void vecadd_sse_wrapper(const T* x, const T* y, std::size_t n, std::size_t m, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        sgdlib::detail::vecadd_sse_float(x, y, n, m, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        sgdlib::detail::vecadd_sse_double(x, y, n, m, out);
    }
}

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
    vecadd_sse_wrapper<T>(nullptr, nullptr, n8, n8, nullptr);
    vecadd_sse_wrapper<T>(nullptr, y8.data(), n8, n8, out8.data());
    vecadd_sse_wrapper<T>(x8.data(), nullptr, n8, n8, out8.data());
    vecadd_sse_wrapper<T>(x8.data(), y8.data(), n8, n8, nullptr);
    vecadd_sse_wrapper<T>(x8.data(), y8.data(), n8, 7, nullptr);

    // small size n = m = 3 < 4
    std::vector<T> out3(3);
    vecadd_sse_wrapper<T>(x8.data(), y8.data(), 3, 3, out3.data());
    EXPECT_FLOAT_EQ(out3[0], x8[0] + y8[0]);
    EXPECT_FLOAT_EQ(out3[1], x8[1] + y8[1]);
    EXPECT_FLOAT_EQ(out3[2], x8[2] + y8[2]);

    // small size n = m = 5 > 4
    std::vector<T> out5(5);
    vecadd_sse_wrapper<T>(x8.data(), y8.data(), 5, 5, out5.data());
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(out5[i], x8[i] + y8[i]);
    }

    // aligned case n = m = 8
    vecadd_sse_wrapper<T>(x8.data(), y8.data(), 8, 8, out8.data());
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

    vecadd_sse_wrapper<T>(x_large.data(), y_large.data(), 1025, 1025, out1025_1.data());
    sgdlib::detail::vecadd_ansi<T>(x_large, y_large, out1025_2);
    for (std::size_t i = 0; i < 1025; ++i) {
        EXPECT_FLOAT_EQ(out1025_1[i], out1025_2[i]);
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
    vecadd_sse_wrapper<T>(x_huge.data(), y_huge.data(), huge_size, huge_size, out_huge1.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecadd without constant C SIMD execution time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecadd_ansi<T>(x_huge, y_huge, out_huge2);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "vecadd without constant C ANSI execution time: " << elapsed2.count() << " seconds\n";

    for (std::size_t i = 0; i < huge_size; ++i) {
        EXPECT_FLOAT_EQ(out_huge1[i], out_huge2[i]);
    }
}


// ****************************vecadd 2*******************************//
template <typename T>
void vecadd_sse_wrapper(const T* x, const T* y, const T c, std::size_t n, std::size_t m, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        sgdlib::detail::vecadd_sse_float(x, y, c, n, n, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        sgdlib::detail::vecadd_sse_double(x, y, c, n, n, out);
    }
}

TYPED_TEST(MathOpsSSETest, VecAddWithConstCSSETest) {
    using T = typename TestFixture::Type;

    std::vector<T> x8;
    std::vector<T> y8;
    std::vector<T> out8;

    T c = 2.5;
    x8 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    y8 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    out8.resize(8);

    // empty cases
    std::size_t n8 = 8;
    vecadd_sse_wrapper<T>(nullptr, nullptr, c, n8, n8, nullptr);
    vecadd_sse_wrapper<T>(nullptr, y8.data(), c, n8, n8, out8.data());
    vecadd_sse_wrapper<T>(x8.data(), nullptr, c, n8, n8, out8.data());
    vecadd_sse_wrapper<T>(x8.data(), y8.data(), c, n8, n8, nullptr);
    vecadd_sse_wrapper<T>(x8.data(), y8.data(), c, n8, 7, nullptr);

    // small size n = m = 3 < 4
    std::vector<T> out3(3);
    vecadd_sse_wrapper<T>(x8.data(), y8.data(), c, 3, 3, out3.data());
    EXPECT_FLOAT_EQ(out3[0], x8[0] * c + y8[0]);
    EXPECT_FLOAT_EQ(out3[1], x8[1] * c + y8[1]);
    EXPECT_FLOAT_EQ(out3[2], x8[2] * c + y8[2]);

    // small size n = m = 5 > 4
    std::vector<T> out5(5);
    vecadd_sse_wrapper<T>(x8.data(), y8.data(), c, 5, 5, out5.data());
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(out5[i], x8[i] * c + y8[i]);
    }

    // aligned case n = m = 8
    vecadd_sse_wrapper<T>(x8.data(), y8.data(), c, 8, 8, out8.data());
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

    vecadd_sse_wrapper<T>(x_large.data(), y_large.data(), c, 1025, 1025, out1025_1.data());
    sgdlib::detail::vecadd_ansi<T>(x_large, y_large, c, out1025_2);
    for (std::size_t i = 0; i < 1025; ++i) {
        EXPECT_FLOAT_EQ(out1025_1[i], out1025_2[i]);
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
    vecadd_sse_wrapper<T>(x_huge.data(), y_huge.data(), c, huge_size, huge_size, out_huge1.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecadd with constant C SIMD execution time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecadd_ansi<T>(x_huge, y_huge, c, out_huge2);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "vecadd with constant C ANSI execution time: " << elapsed2.count() << " seconds\n";

    for (std::size_t i = 0; i < huge_size; ++i) {
        EXPECT_FLOAT_EQ(out_huge1[i], out_huge2[i]);
    }

}

// ****************************vecdiff**********************************//
template <typename T>
void vecdiff_sse_wrapper(const T* x, const T* y, std::size_t n, std::size_t m, T* out) {
    if constexpr (std::is_same_v<T, float>) {
        sgdlib::detail::vecdiff_sse_float(x, y, n, m, out);
    }
    else if constexpr (std::is_same_v<T, double>) {
        sgdlib::detail::vecdiff_sse_double(x, y, n, m, out);
    }
}

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
    vecdiff_sse_wrapper<T>(nullptr, nullptr, n8, n8, nullptr);
    vecdiff_sse_wrapper<T>(nullptr, y8.data(), n8, n8, out8.data());
    vecdiff_sse_wrapper<T>(x8.data(), nullptr, n8, n8, out8.data());
    vecdiff_sse_wrapper<T>(x8.data(), y8.data(), n8, n8, nullptr);
    vecdiff_sse_wrapper<T>(x8.data(), y8.data(), n8, 7, nullptr);

    // small size n = m = 3 < 4
    std::vector<T> out3(3);
    vecdiff_sse_wrapper<T>(x8.data(), y8.data(), 3, 3, out3.data());
    EXPECT_FLOAT_EQ(out3[0], x8[0] - y8[0]);
    EXPECT_FLOAT_EQ(out3[1], x8[1] - y8[1]);
    EXPECT_FLOAT_EQ(out3[2], x8[2] - y8[2]);

    // small size n = m = 5 > 4
    std::vector<T> out5(5);
    vecdiff_sse_wrapper<T>(x8.data(), y8.data(), 5, 5, out5.data());
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(out5[i], x8[i] - y8[i]);
    }

    // aligned case n = m = 8
    vecdiff_sse_wrapper<T>(x8.data(), y8.data(), 8, 8, out8.data());
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

    vecdiff_sse_wrapper<T>(x_large.data(), y_large.data(), 1025, 1025, out1025_1.data());
    sgdlib::detail::vecdiff_ansi<T>(x_large, y_large, out1025_2);
    for (std::size_t i = 0; i < 1025; ++i) {
        EXPECT_FLOAT_EQ(out1025_1[i], out1025_2[i]);
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
    vecdiff_sse_wrapper<T>(x_huge.data(), y_huge.data(), huge_size, huge_size, out_huge1.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "vecdiff SIMD execution time: " << elapsed1.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    sgdlib::detail::vecdiff_ansi<T>(x_huge, y_huge, out_huge2);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "vecdiff ANSI execution time: " << elapsed2.count() << " seconds\n";

    for (std::size_t i = 0; i < huge_size; ++i) {
        EXPECT_FLOAT_EQ(out_huge1[i], out_huge2[i]);
    }
}


// ****************************vecdot**********************************//
template <typename T>
T vecdot_sse_wrapper(const T* x, const T* y, std::size_t n, std::size_t m) {
    T prod;
    if constexpr (std::is_same_v<T, float>) {
        prod = sgdlib::detail::vecdot_sse_float(x, y, n, m);
    }
    else if constexpr (std::is_same_v<T, double>) {
        prod = sgdlib::detail::vecdot_sse_double(x, y, n, m);
    }
    return prod;
}

TYPED_TEST(MathOpsSSETest, VecDotSSETest) {
    using T = typename TestFixture::Type;

    // empty x / y
    std::vector<T> x1 = {1.0, 2.0};
    EXPECT_FLOAT_EQ(vecdot_sse_wrapper<T>(nullptr, x1.data(), 2, 2), 0.0);
    EXPECT_FLOAT_EQ(vecdot_sse_wrapper<T>(x1.data(), nullptr, 2, 2), 0.0);

    // x length != y length
    std::vector<T> x2 = {1.0, 2.0};
    std::vector<T> y2 = {1.0, 2.0, 3.0};
    EXPECT_FLOAT_EQ(vecdot_sse_wrapper<T>(x2.data(), y2.data(), 2, 3), 0.0);

    // small size
    std::vector<T> x3 = {1.0, 2.0, 3.0};  // n=3
    std::vector<T> y3 = {4.0, 5.0, 6.0};
    // 1.0*4.0 + 2.0*5.0 + 3.0*6.0;
    T expected3 = sgdlib::detail::vecdot_ansi<T>(x3, y3);
    EXPECT_FLOAT_EQ(vecdot_sse_wrapper<T>(x3.data(), y3.data(), 3, 3), expected3);

    // unaligned small size
    std::vector<T> x4 = {1.0, 2.0, 3.0, 4.0, 5.0};  // n=5
    std::vector<T> y4 = {2.0, 3.0, 4.0, 5.0, 6.0};
    T expected4 = sgdlib::detail::vecdot_ansi<T>(x4, y4);;
    EXPECT_FLOAT_EQ(vecdot_sse_wrapper<T>(x4.data(), y4.data(), 5, 5), expected4);

    // unaligned big size 1024
    std::size_t size1 = 1025;
    std::vector<T> x5(size1), y5(size1);
    x5 = this->generate_test_data(size1, false, 1, -1);
    y5 = this->generate_test_data(size1, false, 1, -1);

    T expected5 = sgdlib::detail::vecdot_ansi<T>(x5, y5);
    T result5 = vecdot_sse_wrapper<T>(x5.data(), y5.data(), size1, size1);
    EXPECT_NEAR(result5, expected5, 1e-4);

    // performance test
    const std::size_t size2 = 1 << 20;  // 1M elements
    std::vector<T> x6(size2), y6(size2);
    x6 = this->generate_test_data(size2, false, 1, -1);
    y6 = this->generate_test_data(size2, false, 1, -1);

    auto start = std::chrono::high_resolution_clock::now();
    T result6 = vecdot_sse_wrapper<T>(x6.data(), y6.data(), size2, size2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "vecdot SIMD execution time: " << elapsed.count() << " s\n";

    start = std::chrono::high_resolution_clock::now();
    T expected6 = sgdlib::detail::vecdot_ansi<T>(x6, y6);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "vecdot ANSI execution time: " << elapsed.count() << " s\n";

    EXPECT_NEAR(result6, expected6, 1e-2);
}


