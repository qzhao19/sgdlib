#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/core/decay_policy/base.hpp"
#include "sgdlib/core/decay_policy/exponential_decay.hpp"
#include "sgdlib/core/decay_policy/invscaling_decay.hpp"

namespace sgdlib {

class LRDecayTest : public ::testing::Test {
public:
    virtual void SetUp(std::string lr_decay_policy) {
        double eta0 = 0.0001, decay = 0.0025;

        if (lr_decay_policy == "invscaling") {
            lr_decay = new sgdlib::internal::InvscalingDecay(eta0, decay);
        }
        else if (lr_decay_policy == "exponential") {
            lr_decay = new sgdlib::internal::ExponentialDecay(eta0, decay);
        }
    }

    virtual void TearDown() {
        if (lr_decay) {
            delete lr_decay;
            lr_decay = nullptr;
        }
    }
    sgdlib::internal::LRDecay* lr_decay;
};

TEST_F(LRDecayTest, InvscalingDecayTest) {
    SetUp("invscaling");
    double eta = lr_decay->compute(10);
    std::cout << eta << std::endl;
}

TEST_F(LRDecayTest, ExponentialDecayTest) {
    SetUp("exponential");
    double eta = lr_decay->compute(10);
    std::cout << eta << std::endl;
}

}