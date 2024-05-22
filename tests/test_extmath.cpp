#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/math/extmath.hpp"

namespace sgdlib {

TEST(ExtmathTest, SigmoidTest){
    double x = sgdlib::internal::sigmoid<double>(10.0);
    std::cout << x << std::endl;
    EXPECT_DOUBLE_EQ(x, 0.9999546021312976);
};
 
}

