#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <type_traits>

#include "sgdlib/container/aligned_vector.hpp"

template <typename T>
class AlignedVectorTest : public ::testing::Test {
protected:
    using VectorType = sgdlib::vector<T>;
};

// Test types
using TestValueTypes = ::testing::Types<int, double, float>;
TYPED_TEST_SUITE(AlignedVectorTest, TestValueTypes);

TYPED_TEST(AlignedVectorTest, DefaultConstructor) {
    using VectorType = typename TestFixture::VectorType;
    VectorType vec;
    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0);
    EXPECT_EQ(vec.capacity(), 0);
    EXPECT_TRUE(vec.is_aligned());
}

TYPED_TEST(AlignedVectorTest, CopyConstructor) {
    using VectorType = typename TestFixture::VectorType;
    using ValueType = typename VectorType::value_type;

    VectorType original = {static_cast<ValueType>(1),
                           static_cast<ValueType>(2),
                           static_cast<ValueType>(3)};
    VectorType copy_v(original);

    EXPECT_EQ(copy_v.size(), original.size());
    EXPECT_TRUE(copy_v.is_aligned());
    for (std::size_t i = 0; i < original.size(); ++i) {
        EXPECT_FLOAT_EQ(copy_v[i], original[i]);
    }
}


TYPED_TEST(AlignedVectorTest, MoveConstructor) {
    using VectorType = typename TestFixture::VectorType;
    using ValueType = typename VectorType::value_type;

    VectorType src{static_cast<ValueType>(1),
                    static_cast<ValueType>(2),
                    static_cast<ValueType>(3)};
    VectorType vec(std::move(src));
    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], 1);
    EXPECT_TRUE(vec.is_aligned());
    EXPECT_TRUE(src.empty());
    EXPECT_EQ(src.size(), 0);
}


TYPED_TEST(AlignedVectorTest, InitializerListConstructor) {
    using VectorType = typename TestFixture::VectorType;
    using ValueType = typename VectorType::value_type;

    ValueType v1 = static_cast<ValueType>(1);
    ValueType v2 = static_cast<ValueType>(2);
    ValueType v3 = static_cast<ValueType>(3);

    VectorType vec = {v1, v2, v3};
    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], v1);
    EXPECT_EQ(vec[1], v2);
    EXPECT_EQ(vec[2], v3);
    EXPECT_TRUE(vec.is_aligned());
}

TYPED_TEST(AlignedVectorTest, IteratorRangeConstructor) {
    using VectorType = typename TestFixture::VectorType;
    using ValueType = typename VectorType::value_type;

    std::vector<ValueType> src = {1, 2, 3};
    VectorType vec(src.begin(), src.end());

    EXPECT_EQ(vec.size(), 3);
    EXPECT_EQ(vec[0], src[0]);
    EXPECT_EQ(vec[1], src[1]);
    EXPECT_EQ(vec[2], src[2]);
    EXPECT_TRUE(vec.is_aligned());
}


TYPED_TEST(AlignedVectorTest, ContiguousMemoryConstructor) {
    using VectorType = typename TestFixture::VectorType;
    using ValueType = typename VectorType::value_type;

    ValueType arr[] = {7, 8, 9};
    VectorType v(std::begin(arr), std::end(arr));

    EXPECT_EQ(v.size(), 3);
    EXPECT_TRUE(v.is_aligned());
    EXPECT_FLOAT_EQ(v[0], 7.0f);
    EXPECT_FLOAT_EQ(v[1], 8.0f);
    EXPECT_FLOAT_EQ(v[2], 9.0f);
}

TYPED_TEST(AlignedVectorTest, EmptyContiguousMemoryConstructor) {
    using VectorType = typename TestFixture::VectorType;
    using ValueType = typename VectorType::value_type;

    ValueType src[] = {};
    VectorType vec(src, src); // empty range

    EXPECT_TRUE(vec.empty());
    EXPECT_EQ(vec.size(), 0);
    EXPECT_TRUE(vec.is_aligned());
}

TYPED_TEST(AlignedVectorTest, PushBackAndAlignment) {
    using VectorType = typename TestFixture::VectorType;
    using ValueType = typename VectorType::value_type;

    VectorType v;
    for (int i = 0; i < 100; ++i) {
        v.push_back(static_cast<ValueType>(i));
        EXPECT_TRUE(v.is_aligned());
    }

    EXPECT_EQ(v.size(), 100);
    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(v[i], static_cast<ValueType>(i));
    }
}

TYPED_TEST(AlignedVectorTest, ContinuousPushBack) {
    using VectorType = typename TestFixture::VectorType;
    using ValueType = typename VectorType::value_type;
    constexpr size_t N = 5'000'000;  // 500w

    VectorType v;
    v.reserve(N);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        v.push_back(static_cast<ValueType>(i));
        if (i % 100'000 == 0) {
            EXPECT_TRUE(v.is_aligned());
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Time for " << N << " push_back operations: "
              << duration.count() << " seconds\n";

    EXPECT_EQ(v.size(), N);
    EXPECT_TRUE(v.is_aligned());
}


TYPED_TEST(AlignedVectorTest, ReserveAndAlignment) {
    using VectorType = typename TestFixture::VectorType;
    using ValueType = typename VectorType::value_type;

    VectorType v;
    v.reserve(100);
    EXPECT_GE(v.capacity(), 100);
    EXPECT_TRUE(v.is_aligned());

    ValueType val = static_cast<ValueType>(42);
    v.push_back(val);
    EXPECT_TRUE(v.is_aligned());
    EXPECT_FLOAT_EQ(v[0], static_cast<ValueType>(42));
}

TYPED_TEST(AlignedVectorTest, LargeSizeConstructor) {
    using VectorType = typename TestFixture::VectorType;
    using ValueType = typename VectorType::value_type;

    constexpr size_t N = 1'000'000;  // 100w
    std::vector<double> source(N);

    for (size_t i = 0; i < N; ++i) {
        source[i] = static_cast<ValueType>(i);
    }

    // performance test
    auto start = std::chrono::high_resolution_clock::now();
    VectorType vec(source.data(), source.data() + N);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Construction time for " << N
              << " elements: " << duration.count() << " seconds\n";


    EXPECT_EQ(vec.size(), N);
    EXPECT_TRUE(vec.is_aligned());

    // sampling test
    const std::size_t sample_points[] = {0, N/4, N/2, 3*N/4, N-1};
    for (std::size_t pos : sample_points) {
        EXPECT_DOUBLE_EQ(vec[pos], source[pos]);
    }


    std::size_t expected_bytes = N * sizeof(ValueType);
    EXPECT_GE(vec.capacity() * sizeof(ValueType), expected_bytes);

    auto modify_start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < N; i += 1000) {
        vec[i] *= 2.0;
    }
    auto modify_end = std::chrono::high_resolution_clock::now();

    duration = modify_end - modify_start;
    std::cout << "Modification time: " << duration.count() << " seconds\n";

    for (std::size_t pos : sample_points) {
        if (pos % 1000 == 0) {
            EXPECT_DOUBLE_EQ(vec[pos], source[pos] * 2.0);
        } else {
            EXPECT_DOUBLE_EQ(vec[pos], source[pos]);
        }
    }
}

TYPED_TEST(AlignedVectorTest, ExtremeLargeCapacity) {
    constexpr size_t N = 100'000'000;  // 100 million
    using VectorType = typename TestFixture::VectorType;
    using ValueType = typename VectorType::value_type;

    VectorType v;
    v.reserve(N);

    EXPECT_GE(v.capacity(), N);
    EXPECT_TRUE(v.is_aligned());
    EXPECT_TRUE(v.empty());
}

TYPED_TEST(AlignedVectorTest, DifferentButCompatibleType) {
    constexpr size_t N = 4;
    using VectorType = typename TestFixture::VectorType;
    using ValueType = typename VectorType::value_type;

    ValueType source[N] = {1, 2, 3, 4};

    // if trigger static_assert
    // vector<float> v(source, source + N);  // should not compile

    // correct way to construct vector with different but compatible type
    VectorType v(source, source + N);
    EXPECT_EQ(v.size(), N);
    EXPECT_TRUE(v.is_aligned());
    for (size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(v[i], source[i]);
    }
}

TYPED_TEST(AlignedVectorTest, ElementArithmeticOperations) {
    constexpr size_t N = 10'000'000;
    using VectorType = typename TestFixture::VectorType;
    using ValueType = typename VectorType::value_type;

    VectorType a = {1, 2, 3};
    VectorType b = {4, 5, 6};
    VectorType result(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }

    EXPECT_EQ(result.size(), 3);
    EXPECT_DOUBLE_EQ(result[0], 5);
    EXPECT_DOUBLE_EQ(result[1], 7);
    EXPECT_DOUBLE_EQ(result[2], 9);
    EXPECT_TRUE(result.is_aligned());

    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                  [](ValueType x, ValueType y) { return x - y; });

    EXPECT_EQ(result.size(), 3);
    EXPECT_FLOAT_EQ(result[0], -3);
    EXPECT_FLOAT_EQ(result[1], -3);
    EXPECT_FLOAT_EQ(result[2], -3);
    EXPECT_TRUE(result.is_aligned());

    std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                  [](ValueType x, ValueType y) { return x * y; });

    EXPECT_EQ(result.size(), 3);
    EXPECT_FLOAT_EQ(result[0], 4);
    EXPECT_FLOAT_EQ(result[1], 10);
    EXPECT_FLOAT_EQ(result[2], 18);
    EXPECT_TRUE(result.is_aligned());
}

TYPED_TEST(AlignedVectorTest, PerformanceComparison) {
    constexpr size_t N = 10'000'000;
    using VectorType = typename TestFixture::VectorType;
    using ValueType = typename VectorType::value_type;

    std::vector<ValueType> std_a(N), std_b(N), std_result(N);
    VectorType aligned_a(N), aligned_b(N), aligned_result(N);

    for (size_t i = 0; i < N; ++i) {
        std_a[i] = aligned_a[i] = i * static_cast<ValueType>(2);
        std_b[i] = aligned_b[i] = i * static_cast<ValueType>(3);
    }

    // stl vector performance
    auto std_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        std_result[i] = std_a[i] + std_b[i];
    }
    auto std_end = std::chrono::high_resolution_clock::now();

    // AlignedVector performance
    auto aligned_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        aligned_result[i] = aligned_a[i] + aligned_b[i];
    }
    auto aligned_end = std::chrono::high_resolution_clock::now();

    auto std_duration = std::chrono::duration<double>(std_end - std_start);
    auto aligned_duration = std::chrono::duration<double>(aligned_end - aligned_start);

    std::cout << "\nPerformance Comparison (N=" << N << "):\n";
    std::cout << "std::vector time: " << std_duration.count() << " seconds\n";
    std::cout << "AlignedVector time: " << aligned_duration.count() << " seconds\n";
    std::cout << "Speedup: " << std_duration.count() / aligned_duration.count() << "x\n";

    EXPECT_TRUE(aligned_result.is_aligned());
    // check result
    for (size_t i = 0; i < 10; ++i) {
        size_t idx = i * (N / 10);
        EXPECT_DOUBLE_EQ(std_result[idx], aligned_result[idx]);
    }
}


