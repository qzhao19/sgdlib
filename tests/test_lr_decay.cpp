#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

// include private header file
#include "sgdlib/core/lr_decay.hpp"

class LRDecayTest : public ::testing::Test {
public:
    void SetUp(std::string lr_decay_policy) {
        double eta0 = 0.0001, gamma = 0.0025;
        LRDecayParamType lr_decay_param = {{"eta0", eta0}, {"gamma", gamma}};
        lr_decay = sgdlib::detail::LRDecayRegistry()->Create(lr_decay_policy, lr_decay_param);
    }

    std::unique_ptr<sgdlib::detail::LRDecay> lr_decay; 
};

TEST_F(LRDecayTest, InvscalingDecayTest) {
    SetUp("Invscaling");
    double eta = lr_decay->compute(10);
    std::cout << eta << std::endl;
    EXPECT_DOUBLE_EQ(eta, 9.940231944093156e-05);
}

TEST_F(LRDecayTest, ExponentialDecayTest) {
    SetUp("Exponential");
    double eta = lr_decay->compute(10);
    std::cout << eta << std::endl;
    EXPECT_DOUBLE_EQ(eta, 9.7530991202833268e-05);
}
