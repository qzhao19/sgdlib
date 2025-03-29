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
        LossParamType loss_params = {{"alpha", 0.0}};
        loss_fn = LossFunctionRegistry()->Create(loss, loss_params);
    }

    virtual void TearDown() {}

    std::shared_ptr<sgdlib::LossFunction> loss_fn; 
};

TEST_F(LossTest, LogLossTest) {
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
    for (std::size_t i = 0; i < g.size(); ++i) {
        EXPECT_NEAR(g[i], expect_g[i], tol);
    }
};


TEST_F(LossTest, LogLossAllDataTest) {
    SetUp("LogLoss");
    std::vector<double> X_train = {
        5.1, 3.5, 1.4, 0.2,4.9, 3. , 1.4, 0.2,4.7, 3.2, 1.3, 0.2,4.6, 3.1, 1.5, 0.2,
        5. , 3.6, 1.4, 0.2,5.4, 3.9, 1.7, 0.4,4.6, 3.4, 1.4, 0.3,5. , 3.4, 1.5, 0.2,
        4.4, 2.9, 1.4, 0.2,4.9, 3.1, 1.5, 0.1,5.4, 3.7, 1.5, 0.2,4.8, 3.4, 1.6, 0.2,
        4.8, 3. , 1.4, 0.1,4.3, 3. , 1.1, 0.1,5.8, 4. , 1.2, 0.2,5.7, 4.4, 1.5, 0.4,
        5.4, 3.9, 1.3, 0.4,5.1, 3.5, 1.4, 0.3,5.7, 3.8, 1.7, 0.3,5.1, 3.8, 1.5, 0.3,
        5.4, 3.4, 1.7, 0.2,5.1, 3.7, 1.5, 0.4,4.6, 3.6, 1. , 0.2,5.1, 3.3, 1.7, 0.5,
        4.8, 3.4, 1.9, 0.2,5. , 3. , 1.6, 0.2,5. , 3.4, 1.6, 0.4,5.2, 3.5, 1.5, 0.2,
        5.2, 3.4, 1.4, 0.2,4.7, 3.2, 1.6, 0.2,4.8, 3.1, 1.6, 0.2,5.4, 3.4, 1.5, 0.4,
        5.2, 4.1, 1.5, 0.1,5.5, 4.2, 1.4, 0.2,4.9, 3.1, 1.5, 0.2,5. , 3.2, 1.2, 0.2,
        5.5, 3.5, 1.3, 0.2,4.9, 3.6, 1.4, 0.1,4.4, 3. , 1.3, 0.2,5.1, 3.4, 1.5, 0.2,
        5. , 3.5, 1.3, 0.3,4.5, 2.3, 1.3, 0.3,4.4, 3.2, 1.3, 0.2,5. , 3.5, 1.6, 0.6,
        5.1, 3.8, 1.9, 0.4,4.8, 3. , 1.4, 0.3,5.1, 3.8, 1.6, 0.2,4.6, 3.2, 1.4, 0.2,
        5.3, 3.7, 1.5, 0.2,5. , 3.3, 1.4, 0.2,7. , 3.2, 4.7, 1.4,6.4, 3.2, 4.5, 1.5,
        6.9, 3.1, 4.9, 1.5,5.5, 2.3, 4. , 1.3,6.5, 2.8, 4.6, 1.5,5.7, 2.8, 4.5, 1.3,
        6.3, 3.3, 4.7, 1.6,4.9, 2.4, 3.3, 1. ,6.6, 2.9, 4.6, 1.3,5.2, 2.7, 3.9, 1.4,
        5. , 2. , 3.5, 1. ,5.9, 3. , 4.2, 1.5,6. , 2.2, 4. , 1. ,6.1, 2.9, 4.7, 1.4,
        5.6, 2.9, 3.6, 1.3,6.7, 3.1, 4.4, 1.4,5.6, 3. , 4.5, 1.5,5.8, 2.7, 4.1, 1. ,
        6.2, 2.2, 4.5, 1.5,5.6, 2.5, 3.9, 1.1,5.9, 3.2, 4.8, 1.8,6.1, 2.8, 4. , 1.3,
        6.3, 2.5, 4.9, 1.5,6.1, 2.8, 4.7, 1.2,6.4, 2.9, 4.3, 1.3,6.6, 3. , 4.4, 1.4,
        6.8, 2.8, 4.8, 1.4,6.7, 3. , 5. , 1.7,6. , 2.9, 4.5, 1.5,5.7, 2.6, 3.5, 1. ,
        5.5, 2.4, 3.8, 1.1,5.5, 2.4, 3.7, 1. ,5.8, 2.7, 3.9, 1.2,6. , 2.7, 5.1, 1.6,
        5.4, 3. , 4.5, 1.5,6. , 3.4, 4.5, 1.6,6.7, 3.1, 4.7, 1.5,6.3, 2.3, 4.4, 1.3,
        5.6, 3. , 4.1, 1.3,5.5, 2.5, 4. , 1.3,5.5, 2.6, 4.4, 1.2,6.1, 3. , 4.6, 1.4,
        5.8, 2.6, 4. , 1.2,5. , 2.3, 3.3, 1. ,5.6, 2.7, 4.2, 1.3,5.7, 3. , 4.2, 1.2,
        5.7, 2.9, 4.2, 1.3,6.2, 2.9, 4.3, 1.3,5.1, 2.5, 3. , 1.1,5.7, 2.8, 4.1, 1.3,
        6.3, 3.3, 6. , 2.5,5.8, 2.7, 5.1, 1.9,7.1, 3. , 5.9, 2.1,6.3, 2.9, 5.6, 1.8,
        6.5, 3. , 5.8, 2.2,7.6, 3. , 6.6, 2.1,4.9, 2.5, 4.5, 1.7,7.3, 2.9, 6.3, 1.8,
        6.7, 2.5, 5.8, 1.8,7.2, 3.6, 6.1, 2.5,6.5, 3.2, 5.1, 2. ,6.4, 2.7, 5.3, 1.9,
        6.8, 3. , 5.5, 2.1,5.7, 2.5, 5. , 2. ,5.8, 2.8, 5.1, 2.4,6.4, 3.2, 5.3, 2.3,
        6.5, 3. , 5.5, 1.8,7.7, 3.8, 6.7, 2.2,7.7, 2.6, 6.9, 2.3,6. , 2.2, 5. , 1.5,
        6.9, 3.2, 5.7, 2.3,5.6, 2.8, 4.9, 2. ,7.7, 2.8, 6.7, 2. ,6.3, 2.7, 4.9, 1.8,
        6.7, 3.3, 5.7, 2.1,7.2, 3.2, 6. , 1.8,6.2, 2.8, 4.8, 1.8,6.1, 3. , 4.9, 1.8,
        6.4, 2.8, 5.6, 2.1,7.2, 3. , 5.8, 1.6,7.4, 2.8, 6.1, 1.9,7.9, 3.8, 6.4, 2. ,
        6.4, 2.8, 5.6, 2.2,6.3, 2.8, 5.1, 1.5,6.1, 2.6, 5.6, 1.4,7.7, 3. , 6.1, 2.3,
        6.3, 3.4, 5.6, 2.4,6.4, 3.1, 5.5, 1.8,6. , 3. , 4.8, 1.8,6.9, 3.1, 5.4, 2.1,
        6.7, 3.1, 5.6, 2.4,6.9, 3.1, 5.1, 2.3,5.8, 2.7, 5.1, 1.9,6.8, 3.2, 5.9, 2.3,
        6.7, 3.3, 5.7, 2.5,6.7, 3. , 5.2, 2.3,6.3, 2.5, 5. , 1.9,6.5, 3. , 5.2, 2. ,
        6.2, 3.4, 5.4, 2.3,5.9, 3. , 5.1, 1.8};
    std::vector<long> y_train = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    
    std::vector<double> w = {1.0, 1.0, 1.0, 1.0};
    std::size_t num_samples = y_train.size();
    std::size_t num_features = w.size();

    double y_hat;
    double loss = 0.0;
    std::vector<double> grad(num_features);
    for (std::size_t i = 0; i < num_samples; ++i) {
        y_hat = std::inner_product(&X_train[i * num_features], 
                                   &X_train[(i + 1) * num_features], 
                                   w.begin(), 0.0);
        
        loss += loss_fn->evaluate(y_hat, y_train[i]);
        for (std::size_t j = 0; j < num_features; ++j) {
            grad[j] += loss_fn->derivate(y_hat, y_train[i]) * X_train[i * num_features + j];
        }
    }
    loss /= static_cast<FeatValType>(num_samples);
    std::transform(grad.begin(), grad.end(), grad.begin(),
                  [num_samples](FeatValType val) { 
                    return val / static_cast<FeatValType>(num_samples); 
                });
    double tolerance = 1e-5;
    std::vector<double> expect_grad = {2.75991, 1.64394, 1.26731, 0.324662};

    EXPECT_NEAR(loss, 5.99602, tolerance);
    for (std::size_t i = 0; i < grad.size(); ++i) {
        EXPECT_NEAR(grad[i], expect_grad[i], tolerance);
    } 
}

}
