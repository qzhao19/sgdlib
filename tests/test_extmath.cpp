#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/common/predefs.hpp"
#include "sgdlib/math/extmath.hpp"

namespace sgdlib {

TEST(ExtmathTest, VectorClipTest){
    std::vector<double> x = {5.2, 3.3, 1.2, 0.3, 
                             4.8, 3.1, 1.6, 0.2};
    sgdlib::internal::clip<double>(x, 2.0, 4.0);
    std::vector<double> expect = {4.0, 3.3, 2.0, 2.0,
                                  4.0, 3.1, 2.0, 2.0};
    EXPECT_EQ(x, expect);
};

TEST(ExtmathTest, ScalarClipTest){
    double a = 5.2;
    sgdlib::internal::clip<double>(a, 2.0, 4.0);
    EXPECT_DOUBLE_EQ(a, 4.0);
};

TEST(ExtmathTest, VectorIsinfTest){
    double a = INF;
    std::vector<double> x1 = {5.2, 3.3, a, 0.3};
    bool has_inf = sgdlib::internal::isinf<double>(x1);
    EXPECT_TRUE(has_inf);

    std::vector<double> x2 = {5.2, 3.3, 2.6, 0.3};
    bool no_inf = sgdlib::internal::isinf<double>(x2);
    EXPECT_FALSE(no_inf);
};

TEST(ExtmathTest, ScalarIsinfTest){
    double a = INF;
    bool is_inf = sgdlib::internal::isinf<double>(a);
    EXPECT_TRUE(is_inf);
};

TEST(ExtmathTest, Sqnorm2Test){
    std::vector<double> x = {5.2, 3.3, 2.6, 0.3};
    double norm = sgdlib::internal::sqnorm2<double>(x, true);
    EXPECT_DOUBLE_EQ(norm, 6.691786009728643);
};

TEST(ExtmathTest, Sqnorm1Test){
    std::vector<double> x = {5.2, 3.3, 2.6, 0.3};
    double norm = sgdlib::internal::norm1<double>(x);
    EXPECT_DOUBLE_EQ(norm, 11.4);
};

TEST(ExtmathTest, InplaceVectorScalarDotTest){
    std::vector<double> x = {5.2, 3.3, 2.6, 0.3};
    double c = 10.0;
    sgdlib::internal::dot<double>(x, c);
    std::vector<double> expect = {52.0, 33.0, 26.0, 3.0};
    EXPECT_EQ(x, expect);
};

TEST(ExtmathTest, IteratorVectorScalarDotTest){
    std::vector<double> x = {5.2, 3.3, 2.6, 0.3};
    std::vector<double> out(x.size());
    double c = 10.0;
    sgdlib::internal::dot<double>(x.begin(), x.end(), c, out);
    std::vector<double> expect = {52.0, 33.0, 26.0, 3.0};
    EXPECT_EQ(out, expect);
};

TEST(ExtmathTest, TwoVectorsDotTest){
    std::vector<double> x = {5.2, 3.3, 2.6, 0.3};
    std::vector<double> y = {5.2, 3.3, 2.6, 0.3};
    double out = 0.0;
    sgdlib::internal::dot<double>(x, y, out);
    double expect = 44.78;
    EXPECT_DOUBLE_EQ(out, expect);
};

TEST(ExtmathTest, RowNormsTest){
    std::vector<double> x = {5.2, 3.3, 1.2, 0.3, 
                             6.4, 3.1, 5.5, 1.8, 
                             4.75, 3.1, 1.32, 0.1};
    std::vector<double> out(3);
    bool sq = false;
    sgdlib::internal::row_norms<double>(x, sq, out);
    std::vector<double> expect = {6.28171951, 9.16842407, 5.82450856};
    double tolerance = 1e-8;
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(expect[i], out[i], tolerance);
    }
};

TEST(ExtmathTest, ColNormsTest){
    std::vector<double> x = {5.2, 3.3, 1.2, 0.3, 
                             6.4, 3.1, 5.5, 1.8, 
                             4.75, 3.1, 1.32, 0.1};
    std::vector<double> out(4);
    bool sq = true;
    sgdlib::internal::col_norms<double>(x, sq, out);
    std::vector<double> expect = {90.5625, 30.11, 33.4324, 3.34};
    double tolerance = 1e-8;
    for (size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(expect[i], out[i], tolerance);
    }
};


}

