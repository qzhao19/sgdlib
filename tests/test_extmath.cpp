#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/common/predefs.hpp"
#include "sgdlib/math/extmath.hpp"

namespace sgdlib {

TEST(ExtmathTest, SigmoidTest){
    double x = sgdlib::internal::sigmoid<double>(10.0);
    std::cout << x << std::endl;
    EXPECT_DOUBLE_EQ(x, 0.9999546021312976);
};

TEST(ExtmathTest, ClipTest){
    std::vector<double> x = {5.2, 3.3, 1.2, 0.3, 
                             4.8, 3.1, 1.6, 0.2};
    sgdlib::internal::clip<double>(x, 2.0, 4.0);
    std::vector<double> expect = {4.0, 3.3, 2.0, 2.0,
                                  4.0, 3.1, 2.0, 2.0};
    EXPECT_EQ(x, expect);
};

TEST(ExtmathTest, IsinfTest){

    double a = INF;
    std::vector<double> x1 = {5.2, 3.3, a, 0.3};
    bool has_inf = sgdlib::internal::isinf<double>(x1);
    EXPECT_TRUE(has_inf);

    std::vector<double> x2 = {5.2, 3.3, 2.6, 0.3};
    bool no_inf = sgdlib::internal::isinf<double>(x2);
    EXPECT_FALSE(no_inf);
};

}

