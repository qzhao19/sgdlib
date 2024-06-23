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

TEST_F(LossTest, LogLoss) {
    std::vector<double> x1 = {5.2, 3.3, 1.2, 0.3};
    std::vector<double> x2 = {6.4, 3.1, 5.5, 1.8};
    long y1 = 1, y2 = 1;
    std::vector<double> w = {1.0, 1.0, 1.0, 1.0};

    double y1_pred, y2_pred;
    y1_pred = std::inner_product(x1.begin(), 
                                 x1.end(), 
                                 w.begin(), 0.0);
    y2_pred = std::inner_product(x2.begin(), 
                                 x2.end(), 
                                 w.begin(), 0.0);

    SetUp("LogLoss");
    double loss1 = loss_fn->evaluate(y1_pred, y1);
    EXPECT_DOUBLES_EQUAL(loss1, 4.5398899216870535e-05);

    double dloss1 = loss_fn->derivate(y1_pred, y1);
    EXPECT_DOUBLES_EQUAL(dloss1, -4.5397868702434395e-05);

    double loss2 = loss_fn->evaluate(y2_pred, y2);
    EXPECT_DOUBLES_EQUAL(loss2, 5.0565312213476096e-08);

    double dloss2 = loss_fn->derivate(y2_pred, y2);
    EXPECT_DOUBLES_EQUAL(dloss2, -5.0565310926504413e-08);

    double loss = (loss1 + loss2) / 2.0;
    EXPECT_DOUBLES_EQUAL(loss, 2.272473226451872e-05);

    std::vector<double> g(x1.size()), g1(x1.size()), g2(x1.size());
    for (std::size_t i = 0; i < x1.size(); ++i) {
        g1[i] = x1[i] * dloss1;
        g2[i] = x2[i] * dloss2;
        g[i] = (g1[i] + g2[i]) / 2.0;
    }

    double tol = 1e-6;
    std::vector<double> expect_g = {-1.18196268e-04, -7.49848596e-05, 
                                    -2.73777758e-05, -6.85518909e-06};
    for (size_t i = 0; i < g.size(); ++i) {
        EXPECT_NEAR(g[i], expect_g[i], tol);
    }
};




}
