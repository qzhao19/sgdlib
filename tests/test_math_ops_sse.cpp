#include <gtest/gtest.h>

#include "sgdlib/common/prereqs.hpp"
#include "sgdlib/math/kernels/ops_ansi.hpp"
#include "sgdlib/math/kernels/ops_sse_double.hpp"
#include "sgdlib/math/kernels/ops_sse_float.hpp"

// ****************************VectorClipTest*******************************//
TEST(ClipSSETest, ClipSSEFloatTest) {
    // Test n < 4 cases
    std::vector<float> data1 = {1.5f};
    std::vector<float> expected1 = data1;
    sgdlib::detail::vec_clip_ansi<float>(0.0f, 2.0f, expected1);
    sgdlib::detail::vec_clip_sse_float(0.0f, 2.0f, data1.data(), data1.size());
    EXPECT_EQ(data1, expected1);

    std::vector<float> data2 = {-1.0f, 3.0f, 0.5f};
    std::vector<float> expected2 = data2;
    sgdlib::detail::vec_clip_ansi<float>(0.0f, 1.0f, expected2);
    sgdlib::detail::vec_clip_sse_float(0.0f, 1.0f, data2.data(), data2.size());
    EXPECT_EQ(data2, expected2);

    // Test n = 4 (exactly one SIMD operation)
    std::vector<float> data3 = {-2.0f, 0.5f, 1.5f, 3.0f};
    std::vector<float> expected3 = data3;
    sgdlib::detail::vec_clip_ansi<float>(0.0f, 1.0f, expected3);
    sgdlib::detail::vec_clip_sse_float(0.0f, 1.0f, data3.data(), data3.size());
    EXPECT_EQ(data3, expected3);

    // Test n > 4 with remainder
    std::vector<float> data4(127, 0.0f); // Prime number size
    for (std::size_t i = 0; i < data4.size(); ++i) {
        data4[i] = static_cast<float>(i) - 50.0f;
    }

    std::vector<float> expected4 = data4;
    sgdlib::detail::vec_clip_ansi<float>(-10.0f, 10.0f, expected4);
    sgdlib::detail::vec_clip_sse_float(-10.0f, 10.0f, data4.data(), data4.size());

    for (std::size_t i = 0; i < data4.size(); ++i) {
        EXPECT_FLOAT_EQ(data4[i], expected4[i]) << "Mismatch at index " << i;
    }

    // Test NaN/inf (behavior depends on requirements)
    std::vector<float> specials = {NAN, INFINITY, -INFINITY};
    std::vector<float> spec_expected = specials;
    sgdlib::detail::vec_clip_ansi<float>(0.0f, 1.0f, spec_expected);
    sgdlib::detail::vec_clip_sse_float(0.0f, 1.0f, specials.data(), specials.size());
    for (std::size_t i = 0; i < specials.size(); ++i) {
        EXPECT_EQ(std::isnan(specials[i]), std::isnan(spec_expected[i]));
        if (!std::isnan(specials[i])) {
            EXPECT_FLOAT_EQ(specials[i], spec_expected[i]);
        }
    }
}

TEST(ClipSSETest, ClipSSEDoubleTest) {
    // Test n < 4 cases
    std::vector<double> data1 = {1.5};
    std::vector<double> expected1 = data1;
    sgdlib::detail::vec_clip_ansi<double>(0.0, 2.0, expected1);
    sgdlib::detail::vec_clip_sse_double(0.0, 2.0, data1.data(), data1.size());
    EXPECT_EQ(data1, expected1);

    std::vector<double> data2 = {-1.0, 3.0, 0.5};
    std::vector<double> expected2 = data2;
    sgdlib::detail::vec_clip_ansi<double>(0.0, 1.0, expected2);
    sgdlib::detail::vec_clip_sse_double(0.0, 1.0, data2.data(), data2.size());
    EXPECT_EQ(data2, expected2);

    // Test n = 4 (exactly one SIMD operation)
    std::vector<double> data3 = {-2.0, 0.5, 1.5, 3.0};
    std::vector<double> expected3 = data3;
    sgdlib::detail::vec_clip_ansi<double>(0.0, 1.0, expected3);
    sgdlib::detail::vec_clip_sse_double(0.0, 1.0, data3.data(), data3.size());
    EXPECT_EQ(data3, expected3);

    // Test n > 4 with remainder
    std::vector<double> data4(127, 0.0); // Prime number size
    for (std::size_t i = 0; i < data4.size(); ++i) {
        data4[i] = static_cast<double>(i) - 50.0;
    }

    std::vector<double> expected4 = data4;
    sgdlib::detail::vec_clip_ansi<double>(-10.0, 10.0, expected4);
    sgdlib::detail::vec_clip_sse_double(-10.0, 10.0, data4.data(), data4.size());

    for (std::size_t i = 0; i < data4.size(); ++i) {
        EXPECT_FLOAT_EQ(data4[i], expected4[i]) << "Mismatch at index " << i;
    }

    // Test NaN/inf (behavior depends on requirements)
    std::vector<double> specials = {NAN, INFINITY, -INFINITY};
    std::vector<double> spec_expected = specials;
    sgdlib::detail::vec_clip_ansi<double>(0.0, 1.0, spec_expected);

    for (std::size_t j = 0; j < spec_expected.size(); ++j) {
        std::cout << spec_expected[j] << " ";
    }
    std::cout << std::endl;

    sgdlib::detail::vec_clip_sse_double(0.0, 1.0, specials.data(), specials.size());
    for (std::size_t i = 0; i < specials.size(); ++i) {
        EXPECT_EQ(std::isnan(specials[i]), std::isnan(spec_expected[i]));
        if (!std::isnan(specials[i])) {
            EXPECT_FLOAT_EQ(specials[i], spec_expected[i]);
        }
    }
}

// ****************************IsInfTest*******************************//
template<typename T>
class IsInfSSETest : public ::testing::Test {
protected:
    using Type = T;

    void SetUp() override {
        engine_.seed(42);
    }

    std::vector<T> generate_test_array(std::size_t size, bool with_inf) {
        std::uniform_real_distribution<T> dist(-std::numeric_limits<T>::max() / 2,
                                                std::numeric_limits<T>::max() / 2);
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

template <typename T>
bool isinf_sse_wrapper(const T* x, std::size_t n) {
    if constexpr (std::is_same_v<T, float>) {
        return sgdlib::detail::isinf_sse_float(x, n);
    }
    else if constexpr (std::is_same_v<T, double>) {
        return sgdlib::detail::isinf_sse_double(x, n);
    }
    return false;
}

using TestValueTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(IsInfSSETest, TestValueTypes);
TYPED_TEST(IsInfSSETest, IsInfSSEFloatTest) {
    using T = typename TestFixture::Type;

    T* data1 = nullptr;
    EXPECT_FALSE(isinf_sse_wrapper<T>(data1, 0));

    // // small size array
    std::vector<T> data2 = {1.0, 2.0};
    EXPECT_FALSE(isinf_sse_wrapper<T>(data2.data(), data2.size()));
    std::vector<T> data21 = {1.0, std::numeric_limits<T>::infinity()};
    EXPECT_TRUE(isinf_sse_wrapper<T>(data21.data(), data21.size()));

    // aligned size array
    constexpr std::size_t size3 = 128;
    auto data3 = this->generate_test_array(size3, false);
    EXPECT_FALSE(isinf_sse_wrapper<T>(data3.data(), size3));

    data3.back() = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(isinf_sse_wrapper<T>(data3.data(), size3));

    // unaligned size array
    constexpr std::size_t size4 = 127;
    auto data4 = this->generate_test_array(size4, false);
    EXPECT_FALSE(isinf_sse_wrapper<T>(data4.data(), size4));

    data4[125] = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(isinf_sse_wrapper<T>(data4.data(), size4));

    // large size array
    constexpr std::size_t size5 = 1'000'000;
    // without inf
    auto data5 = this->generate_test_array(size5, false);
    EXPECT_FALSE(isinf_sse_wrapper<T>(data5.data(), size5));

    // with inf
    auto data51 = this->generate_test_array(size5, true);
    EXPECT_TRUE(isinf_sse_wrapper<T>(data51.data(), size5));

    // edge cases
    std::vector<T> data6 = {
        std::numeric_limits<T>::min(),
        std::numeric_limits<T>::max(),
        -std::numeric_limits<T>::max()
    };
    EXPECT_FALSE(isinf_sse_wrapper<T>(data6.data(), data6.size()));
}

