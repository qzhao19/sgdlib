#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/core/loss.hpp"

namespace sgdlib {

#define EXPECT_DOUBLES_EQUAL(expected, actual) \
    EXPECT_LE(std::abs(expected - actual), 1e-6)

using ::testing::DoubleLE;

class LossTest : public ::testing::Test {
public:
    virtual void SetUp(std::string loss) {
        LossParamType loss_param = {{"alpha", 0.0}};
        loss_fn = LossFunctionRegistry()->Create(loss, loss_param);
    }

    virtual void TearDown() {}

    std::unique_ptr<sgdlib::LossFunction> loss_fn; 
};

TEST_F(LossTest, LogLossTest) {
    std::vector<double> X = {5.2, 3.3, 1.2, 0.3,
                            4.8, 3.1 , 1.6, 0.2,
                            4.75, 3.1, 1.32, 0.1,
                            5.9, 2.6, 4.1, 1.2,
                            5.1, 2.2, 3.3, 1.1,
                            5.2, 2.7, 4.1, 1.3,
                            6.6, 3.1, 5.25, 2.2,
                            6.3, 2.5, 5.1, 2.0,
                            6.5, 3.1, 5.2, 2.1};
    std::vector<long> y = {0, 0, 0, 0, 1, 1, 1, 1, 1};
    std::vector<double> w = {0.9781, 0.9711, 0.3962, 0.5209};

    SetUp("LogLoss");
    double loss = loss_fn->evaluate(X, y, w);
    EXPECT_DOUBLE_EQ(loss, 4.0159203584860403);

    std::vector<double> grad(4);
    loss_fn->gradient(X, y, w, grad);

    std::vector<double> expect ={2.2939883890082928, 1.3441739235904036, 
                                 0.9131519457304953, 0.19995927206507597};
    EXPECT_EQ(grad, expect);
};


TEST_F(LossTest, HuberLossTest) {
    std::vector<double> X = {5.2, 3.3, 1.2, 0.3,
                            4.8, 3.1 , 1.6, 0.2,
                            4.75, 3.1, 1.32, 0.1,
                            5.9, 2.6, 4.1, 1.2,
                            5.1, 2.2, 3.3, 1.1,
                            5.2, 2.7, 4.1, 1.3,
                            6.6, 3.1, 5.25, 2.2,
                            6.3, 2.5, 5.1, 2.0,
                            6.5, 3.1, 5.2, 2.1};
    std::vector<long> y = {0, 0, 0, 0, 1, 1, 1, 1, 1};
    std::vector<double> w = {0.9781, 0.9711, 0.3962, 0.5209};

    SetUp("HuberLoss");
    double loss = loss_fn->evaluate(X, y, w);
    EXPECT_DOUBLE_EQ(loss, 4.0159203584860403);

    std::vector<double> grad(4);
    loss_fn->gradient(X, y, w, grad);

    std::vector<double> expect ={2.2939883890082928, 1.3441739235904036, 
                                 0.9131519457304953, 0.19995927206507597};
    EXPECT_EQ(grad, expect);
};

}
