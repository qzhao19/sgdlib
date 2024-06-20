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
    double b = 1.0;

    SetUp("LogLoss");
    loss_fn->set_param("alpha", 1.0);
    loss_fn->set_param("wscale", 1.0);
    double loss = loss_fn->evaluate(X, y, w, b);
    EXPECT_DOUBLE_EQ(loss, 4.7189764781884902);

    double grad_b = 0.0;
    std::vector<double> grad_w(4);
    loss_fn->gradient(X, y, w, b, grad_w, grad_b);
    std::vector<double> expect_w = {4.2504766518182384, 3.2865449134325799, 
                                  1.7056665980952996, 1.2417850159468702};
    EXPECT_EQ(grad_w, expect_w);

    std::cout << grad_b << std::endl;
};

}
