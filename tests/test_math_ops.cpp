#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/common/constants.hpp"
#include "sgdlib/common/prereqs.hpp"

// include private header file
#include "sgdlib/math/math_ops.hpp"

struct MyStruct {
    int a;
    double b;
};

TEST(MathOpsTest, VecAllocTest) {

#if defined(USE_SSE)
    std::size_t n = 16;
    auto arr = sgdlib::detail::vecalloc<MyStruct>(n);
    ASSERT_NE(arr, nullptr);
    // check 16 bytes alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(arr.get()) % 16, 0u);
    arr[0].a = 42;
    arr[0].b = 3.14;
    EXPECT_EQ(arr[0].a, 42);
    EXPECT_DOUBLE_EQ(arr[0].b, 3.14);
#elif defined(USE_AVX)
    std::size_t n = 32;
    auto arr = sgdlib::detail::vecalloc<MyStruct>(n);
    ASSERT_NE(arr, nullptr);
    // check 32 bytes alignment
    EXPECT_EQ(reinterpret_cast<uintptr_t>(arr.get()) % 32, 0u);
    arr[0].a = 42;
    arr[0].b = 3.14;
    EXPECT_EQ(arr[0].a, 42);
    EXPECT_DOUBLE_EQ(arr[0].b, 3.14);
#else
    std::size_t n = 16;
    auto arr = sgdlib::detail::vecalloc<MyStruct>(n);
    ASSERT_NE(arr, nullptr);
    for (std::size_t i = 0; i < n; ++i) {
        arr[i].a = static_cast<int>(i);
        arr[i].b = i * 0.5;
    }
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(arr[i].a, static_cast<int>(i));
        EXPECT_DOUBLE_EQ(arr[i].b, i * 0.5);
    }
#endif
    n = 8;
    auto arr2 = sgdlib::detail::vecalloc<MyStruct>(n);
    ASSERT_NE(arr2, nullptr);
    arr2[0].a = 123;
    arr2[0].b = 456.0;
    EXPECT_EQ(arr2[0].a, 123);
    EXPECT_DOUBLE_EQ(arr2[0].b, 456.0);
}


TEST(MathOpsTest, VecSetTest) {
    std::vector<double> data(20, 0);
    const double scalar = 42.0;

    sgdlib::detail::vecset<double>(data, scalar);

    EXPECT_EQ(data.size(), 20);
    EXPECT_EQ(data[0], scalar);
    EXPECT_EQ(data.back(), scalar);
}


TEST(MathOpsTest, VeccpyTest) {
    std::vector<float> x(26, 1.5f), out(26);
    sgdlib::detail::veccpy<float>(x, out);
    EXPECT_EQ(out[0], 1.5f);
}

TEST(MathOpsTest, VecncpyTest) {
    std::vector<float> x(47, 2.0), out(47);
    sgdlib::detail::vecncpy<float>(x, out);
    EXPECT_EQ(out[1], -2.0);
}

TEST(MathOpsTest, VecclipTest) {
    std::vector<float> data{0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                            0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                            0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    sgdlib::detail::vecclip<float>(data, 1.0, 2.0);
    EXPECT_LE(data[0], 2.0);
}

TEST(MathOpsTest, HasinfTest) {
    std::vector<double> data{1.0, std::numeric_limits<double>::infinity(),
                            0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                            0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                            0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    EXPECT_TRUE(sgdlib::detail::hasinf<double>(data));
}

TEST(MathOpsTest, Vecnorm2Test) {
    std::vector<float> x{0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                         0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                         0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    EXPECT_NEAR(sgdlib::detail::vecnorm2<float>(x), 7.245688, 1e-5);
}

TEST(MathOpsTest, Vecnorm2WithPtrTest) {
    std::vector<float> x{0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                         0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                         0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    EXPECT_NEAR(sgdlib::detail::vecnorm2<float>(x.data(), x.data() + x.size()), 7.245688, 1e-5);
}

TEST(MathOpsTest, Vecnorm1Test) {
    std::vector<double> x{0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                          0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                          0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    EXPECT_DOUBLE_EQ(sgdlib::detail::vecnorm1<double>(x), 27.0);
}

TEST(MathOpsTest, VecscaleTest) {
    std::vector<float> x{0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                         0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                         0.5, 1.5, 2.5, 0.5, 1.5, 2.5}, out(18);
    sgdlib::detail::vecscale<float>(x, 2.0, out);
    EXPECT_FLOAT_EQ(out[1], 3.0);
}

TEST(MathOpsTest, VecscalePointersTest) {
    std::vector<double> data{0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                             0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                             0.5, 1.5, 2.5, 0.5, 1.5, 2.5}, out(18);
    sgdlib::detail::vecscale<double>(&data[0], &data[18], 3.0, out);
    EXPECT_DOUBLE_EQ(out[0], 1.5);
}

TEST(MathOpsTest, VecaddTest) {
    std::vector<float> a{0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    std::vector<float> b{0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5}, out(18);
    sgdlib::detail::vecadd<float>(a, b, out);
    EXPECT_FLOAT_EQ(out[0], 1.0);
}

TEST(MathOpsTest, VecaddScaledTest) {
    std::vector<float> a{1.0, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    std::vector<float> b{2.0, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5}, out(18);
    sgdlib::detail::vecadd<float>(a, b, 0.5, out);
    EXPECT_FLOAT_EQ(out[0], 2.5);  // 1 + 0.5*2 = 2
}

TEST(MathOpsTest, Vecadd1InputScaledTest) {
    std::vector<float> a{1.0, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    std::vector<float> out(18);
    sgdlib::detail::vecadd<float>(a, 0.5, out);
    EXPECT_FLOAT_EQ(out[1], 0.75);  // 1 + 0.5*2 = 2
}

TEST(MathOpsTest, VecaddRangeScaledTest) {
    std::vector<float> a{1.0, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    std::vector<float> out(4, 1.26);
    std::vector<float> expect = {1.76, 2.01, 2.51, 1.51, 2.01};
    sgdlib::detail::vecadd<float>(a.data(), a.data() + 4, 0.5, out);
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out[i], expect[i]) << "at i = " << i;
    }
}

TEST(MathOpsTest, VecdiffTest) {
    std::vector<float> a{4.0, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    std::vector<float> b{2.0, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5}, out(18);
    sgdlib::detail::vecdiff<float>(a, b, out);
    EXPECT_FLOAT_EQ(out[0], 2.0);
}

TEST(MathOpsTest, VecdiffWithScalarTest) {
    float scalar = 2.175;
    std::vector<float> a{4.0, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    std::vector<float> b{2.0, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5}, out(18);
    sgdlib::detail::vecdiff<float>(a, b, scalar, out);
    EXPECT_FLOAT_EQ(out[0], 4.0-2.0*2.175);
}


TEST(MathOpsTest, VecdotTest) {
    std::vector<float> a{4.0, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    std::vector<float> b{2.0, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    EXPECT_DOUBLE_EQ(sgdlib::detail::vecdot<float>(a, b), 60.25);
}

TEST(MathOpsTest, VecdotWithPtrTest) {
    std::vector<float> x{4.0, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    std::vector<float> y{2.0, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    float result = sgdlib::detail::vecdot<float>(x.data(), x.data() + x.size(), y.data());
    EXPECT_FLOAT_EQ(result, 60.25f); // 1*4 + 2*5 + 3*6 = 32
}


TEST(MathOpsTest, VecmulTest) {
    std::vector<float> a{4.0, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5};
    std::vector<float> b{2.0, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5,
                        0.5, 1.5, 2.5, 0.5, 1.5, 2.5}, out(18);
    sgdlib::detail::vecmul<float>(a, b, out);
    EXPECT_FLOAT_EQ(out[0], 8.0);
}

TEST(MathOpsTest, ScalarClipTest){
    double a = 5.2;
    sgdlib::detail::clip<double>(a, 2.0, 4.0);
    EXPECT_DOUBLE_EQ(a, 4.0);
};

TEST(MathOpsTest, ScalarIsinfTest){
    double a = INF;
    bool is_inf = sgdlib::detail::isinf<double>(a);
    EXPECT_TRUE(is_inf);
};

TEST(MathOpsTest, RowNormsTest){
    // std::vector<double> x = {5.2, 3.3, 1.2, 0.3,
    //                          6.4, 3.1, 5.5, 1.8,
    //                          4.75, 3.1, 1.32, 0.1};
    std::vector<double> x = {
        5.2 , 6.4 , 4.75,
        3.3 , 3.1 , 3.1 ,
        1.2 , 5.5 , 1.32,
        0.3 , 1.8 , 0.1
    };
    std::vector<long> y = {1, 1, 1};
    sgdlib::detail::ArrayDatasetType dataset(x, y, 3, 4);
    std::vector<double> out(3);
    bool sq = false;
    sgdlib::detail::row_norms<double>(dataset, sq, out);
    std::vector<double> expect = {6.28171951, 9.16842407, 5.82450856};
    double tolerance = 1e-8;
    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(expect[i], out[i], tolerance);
        // std::cout << out[i] << " ";
    }
    // std::cout << std::endl;
};

TEST(MathOpsTest, ColNormsTest){
    std::vector<double> x = {5.2, 3.3, 1.2, 0.3,
                             6.4, 3.1, 5.5, 1.8,
                             4.75, 3.1, 1.32, 0.1};
    std::vector<double> out(4);
    bool sq = true;
    sgdlib::detail::col_norms<double>(x, sq, out);
    std::vector<double> expect = {90.5625, 30.11, 33.4324, 3.34};
    double tolerance = 1e-8;
    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(expect[i], out[i], tolerance);
    }
};

