#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/core/lr_decay.hpp"


namespace sgdlib {

class LRDecayTest : public ::testing::Test {
public:
    virtual void SetUp(std::string lr_decay_policy) {
        double eta0 = 0.0001, decay = 0.0025;
        lr_decay = LRDecayRegistry()->Create(lr_decay_policy, eta0, decay);
    }

    virtual void TearDown() {}
    std::unique_ptr<sgdlib::LRDecay> lr_decay; 
};

TEST_F(LRDecayTest, InvscalingDecayTest) {
    SetUp("Invscaling");
    double eta = lr_decay->compute(10);
    std::cout << eta << std::endl;
    EXPECT_DOUBLE_EQ(eta, 9.9426007395295659e-05);
}

TEST_F(LRDecayTest, ExponentialDecayTest) {
    SetUp("Exponential");
    double eta = lr_decay->compute(10);
    std::cout << eta << std::endl;
    EXPECT_DOUBLE_EQ(eta, 9.7530991202833268e-05);
}

}