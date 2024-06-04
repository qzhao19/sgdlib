#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/math/random.hpp"

class RandomStateTest : public ::testing::Test {
public:
    virtual void SetUp(unsigned long seed) {
        if (seed == -1) {
            random_state = std::make_unique<sgdlib::internal::RandomState>();
        }
        else {
            random_state = std::make_unique<sgdlib::internal::RandomState>(seed);
        }
    }

    virtual void TearDown() {}
    std::unique_ptr<sgdlib::internal::RandomState> random_state; 
};

TEST_F(RandomStateTest, ShuffleTest) {
    SetUp(-1);
    std::vector<double> x = {0.8, 5.1, 12.6, 8.7};
    random_state->shuffle<double>(x);

    for (int i = 0; i < x.size(); ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;
}
