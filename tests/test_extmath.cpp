#include <gtest/gtest.h>
#include <gmock/gmock.h>

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

}

