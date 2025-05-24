#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/common/constants.hpp"
#include "sgdlib/common/prereqs.hpp"

// include private header file
#include "sgdlib/math/math_ops.hpp"


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
    // for (auto c : out) {
    //     std::cout << c << " ";
    // }
    // std::cout << std::endl;
    // std::cout << out[18] << " " << std::endl;
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
    EXPECT_DOUBLE_EQ(out[0], 2.5);  // 1 + 0.5*2 = 2
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


TEST(MathOpsTest, VectorClipTest){
    std::vector<double> x = {5.2, 3.3, 1.2, 0.3,
                             4.8, 3.1, 1.6, 0.2};
    sgdlib::detail::clip<double>(x, 2.0, 4.0);
    std::vector<double> expect = {4.0, 3.3, 2.0, 2.0,
                                  4.0, 3.1, 2.0, 2.0};
    EXPECT_EQ(x, expect);
};

TEST(MathOpsTest, ScalarClipTest){
    double a = 5.2;
    sgdlib::detail::clip<double>(a, 2.0, 4.0);
    EXPECT_DOUBLE_EQ(a, 4.0);
};

TEST(MathOpsTest, VectorIsinfTest){
    double a = INF;
    std::vector<double> x1 = {5.2, 3.3, a, 0.3};
    bool has_inf = sgdlib::detail::isinf<double>(x1);
    EXPECT_TRUE(has_inf);

    std::vector<double> x2 = {5.2, 3.3, 2.6, 0.3};
    bool no_inf = sgdlib::detail::isinf<double>(x2);
    EXPECT_FALSE(no_inf);
};

TEST(MathOpsTest, ScalarIsinfTest){
    double a = INF;
    bool is_inf = sgdlib::detail::isinf<double>(a);
    EXPECT_TRUE(is_inf);
};

TEST(MathOpsTest, Sqnorm2Test){
    std::vector<double> x = {5.2, 3.3, 2.6, 0.3};
    double norm = sgdlib::detail::sqnorm2<double>(x, true);
    EXPECT_DOUBLE_EQ(norm, 6.691786009728643);
};

TEST(MathOpsTest, Sqnorm1Test){
    std::vector<double> x = {5.2, 3.3, 2.6, 0.3};
    double norm = sgdlib::detail::norm1<double>(x);
    EXPECT_DOUBLE_EQ(norm, 11.4);
};

TEST(MathOpsTest, InplaceVectorScalarDotTest){
    std::vector<double> x = {5.2, 3.3, 2.6, 0.3};
    double c = 10.0;
    sgdlib::detail::dot<double>(x, c);
    std::vector<double> expect = {52.0, 33.0, 26.0, 3.0};
    EXPECT_EQ(x, expect);
};

TEST(MathOpsTest, IteratorVectorScalarDotTest){
    std::vector<double> x = {5.2, 3.3, 2.6, 0.3};
    std::vector<double> out(x.size());
    double c = 10.0;
    sgdlib::detail::dot<double>(x.begin(), x.end(), c, out);
    std::vector<double> expect = {52.0, 33.0, 26.0, 3.0};
    EXPECT_EQ(out, expect);
};

TEST(MathOpsTest, TwoVectorsDotTest){
    std::vector<double> x = {5.2, 3.3, 2.6, 0.3};
    std::vector<double> y = {5.2, 3.3, 2.6, 0.3};
    double out = 0.0;
    sgdlib::detail::dot<double>(x, y, out);
    double expect = 44.78;
    EXPECT_DOUBLE_EQ(out, expect);
};

TEST(MathOpsTest, RowNormsTest){
    std::vector<double> x = {5.2, 3.3, 1.2, 0.3,
                             6.4, 3.1, 5.5, 1.8,
                             4.75, 3.1, 1.32, 0.1};
    std::vector<double> out(3);
    bool sq = false;
    sgdlib::detail::row_norms<double>(x, sq, out);
    std::vector<double> expect = {6.28171951, 9.16842407, 5.82450856};
    double tolerance = 1e-8;
    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(expect[i], out[i], tolerance);
    }
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

