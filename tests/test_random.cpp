#include <gtest/gtest.h>
#include <gmock/gmock.h>

// include private header file
#include "sgdlib/math/random.hpp"

class RandomStateTest : public ::testing::Test {
public:
    virtual void SetUp(unsigned long seed) {
        if (seed == -1) {
            random_state = std::make_unique<sgdlib::detail::RandomState>();
        }
        else {
            random_state = std::make_unique<sgdlib::detail::RandomState>(seed);
        }
    }

    virtual void TearDown() {}
    std::unique_ptr<sgdlib::detail::RandomState> random_state;
};

TEST_F(RandomStateTest, UniformRealTest) {
    SetUp(-1);
    double lower_bound = 0.0, upper_bound = 1.0;
    double x = random_state->uniform_real(lower_bound, upper_bound);

    ASSERT_GE(x, lower_bound) << "x should be greater than lower_bound";
    ASSERT_LT(x, upper_bound) << "x should be strickly less than upper_bound";
};

TEST_F(RandomStateTest, UniformIntTest) {
    SetUp(-1);
    long lower_bound = 0, upper_bound = 5;
    long x = random_state->uniform_int(lower_bound, upper_bound);

    ASSERT_GE(x, lower_bound) << "x should be greater than lower_bound";
    ASSERT_LT(x, upper_bound) << "x should be strickly less than upper_bound";
};

TEST_F(RandomStateTest, ShuffleTest) {
    SetUp(0);
    std::vector<double> x = {0.8, 5.1, 12.6, 8.7};
    std::vector<double> expect = x;
    random_state->shuffle<double>(x);

    ASSERT_NE(x, expect);
};

TEST_F(RandomStateTest, SampleTest) {
    SetUp(-1);
    std::vector<double> x = {0.8, 5.1, 12.6, 8.7};
    std::vector<double> expect = x;
    double elem = random_state->sample<double>(x);

    // check if element is in x
    ASSERT_TRUE(std::find(x.begin(), x.end(), elem) != x.end());

    // check non-repeated
    std::vector<int> elements;
    std::vector<double> x2 = {0.8, 5.1, 12.6, 8.7, 9.5};
    for (int i = 0; i < 5; ++i) {
        double elem2 = random_state->sample<double>(x2);;
        ASSERT_TRUE(std::find(elements.begin(), elements.end(), elem2) == elements.end());
        elements.push_back(elem2);
    }
};


TEST_F(RandomStateTest, RandomIndexTest) {
    SetUp(-1);
    std::size_t lower_bound = 0, upper_bound = 5;
    std::size_t x = random_state->random_index(lower_bound, upper_bound);

    ASSERT_GE(x, lower_bound) << "x should be greater than lower_bound";
    ASSERT_LT(x, upper_bound) << "x should be strickly less than upper_bound";
};

