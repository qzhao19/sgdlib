#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

// include private header file
#include "sgdlib/core/loss.hpp"
#include "data/continuous_dataset.hpp"

#define EXPECT_DOUBLES_EQUAL(expected, actual) \
    EXPECT_LE(std::abs(expected - actual), 1e-6)

using ::testing::DoubleLE;

class LossTest : public ::testing::Test {
public:
    void SetUp(std::string loss) {
        sgdlib::LossParamType loss_params = {{"alpha", 0.0}};
        loss_fn = sgdlib::detail::LossFunctionRegistry()->Create(loss, loss_params);
    }

    std::shared_ptr<sgdlib::detail::LossFunctionType> loss_fn;
};

TEST_F(LossTest, LogLossTest) {
    std::vector<double> x1 = {5.2, 3.3, 1.2, 0.3};
    std::vector<double> x2 = {6.4, 3.1, 5.5, 1.8};
    int y1 = 1, y2 = 1;
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

    std::vector<double> x = {
        5.2, 6.4,
        3.3, 3.1,
        1.2, 5.5,
        0.3, 1.8};
    std::vector<int> y = {1, 1};
    std::vector<double> total_grad(4);

    sgdlib::ArrayDatasetType dataset(x, y, 2, 4);

    double total_loss = loss_fn->evaluate_with_gradient(dataset, w, total_grad);
    total_loss /= 2.0;
    sgdlib::detail::vecscale<sgdlib::FeatureScalarType>(total_grad, 0.5, total_grad);
    EXPECT_DOUBLES_EQUAL(total_loss, 2.272473226451872e-05);
    for (std::size_t i = 0; i < g.size(); ++i) {
        EXPECT_NEAR(total_grad[i], expect_g[i], tol);
    }

};


TEST_F(LossTest, LogLossAllDataTest) {
    SetUp("LogLoss");
    std::vector<double> X_train = {
        5.1, 4.9, 4.7, 4.6, 5. , 5.4, 4.6, 5. , 4.4, 4.9, 5.4, 4.8, 4.8,
        4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5. ,
        5. , 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5. , 5.5, 4.9, 4.4,
        5.1, 5. , 4.5, 4.4, 5. , 5.1, 4.8, 5.1, 4.6, 5.3, 5. , 7. , 6.4,
        6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5. , 5.9, 6. , 6.1, 5.6,
        6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7,
        6. , 5.7, 5.5, 5.5, 5.8, 6. , 5.4, 6. , 6.7, 6.3, 5.6, 5.5, 5.5,
        6.1, 5.8, 5. , 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3,
        6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5,
        7.7, 7.7, 6. , 6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2,
        7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6. , 6.9, 6.7, 6.9, 5.8,
        6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9,
        3.5, 3. , 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3. ,
        3. , 4. , 4.4, 3.9, 3.5, 3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3. ,
        3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 4.1, 4.2, 3.1, 3.2, 3.5, 3.6, 3. ,
        3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3. , 3.8, 3.2, 3.7, 3.3, 3.2, 3.2,
        3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2. , 3. , 2.2, 2.9, 2.9,
        3.1, 3. , 2.7, 2.2, 2.5, 3.2, 2.8, 2.5, 2.8, 2.9, 3. , 2.8, 3. ,
        2.9, 2.6, 2.4, 2.4, 2.7, 2.7, 3. , 3.4, 3.1, 2.3, 3. , 2.5, 2.6,
        3. , 2.6, 2.3, 2.7, 3. , 2.9, 2.9, 2.5, 2.8, 3.3, 2.7, 3. , 2.9,
        3. , 3. , 2.5, 2.9, 2.5, 3.6, 3.2, 2.7, 3. , 2.5, 2.8, 3.2, 3. ,
        3.8, 2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8, 3. , 2.8, 3. ,
        2.8, 3.8, 2.8, 2.8, 2.6, 3. , 3.4, 3.1, 3. , 3.1, 3.1, 3.1, 2.7,
        3.2, 3.3, 3. , 2.5, 3. , 3.4, 3.,
        1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4,
        1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5, 1.7, 1.5, 1. , 1.7, 1.9, 1.6,
        1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.4, 1.3,
        1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7, 4.5,
        4.9, 4. , 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4. , 4.7, 3.6,
        4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4. , 4.9, 4.7, 4.3, 4.4, 4.8, 5. ,
        4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4. , 4.4,
        4.6, 4. , 3.3, 4.2, 4.2, 4.2, 4.3, 3. , 4.1, 6. , 5.1, 5.9, 5.6,
        5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5, 5. , 5.1, 5.3, 5.5,
        6.7, 6.9, 5. , 5.7, 4.9, 6.7, 4.9, 5.7, 6. , 4.8, 4.9, 5.6, 5.8,
        6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.1,
        5.9, 5.7, 5.2, 5. , 5.2, 5.4, 5.1,
        0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1,
        0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2,
        0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2,
        0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.4, 1.5,
        1.5, 1.3, 1.5, 1.3, 1.6, 1. , 1.3, 1.4, 1. , 1.5, 1. , 1.4, 1.3,
        1.4, 1.5, 1. , 1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7,
        1.5, 1. , 1.1, 1. , 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3, 1.2,
        1.4, 1.2, 1. , 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.5, 1.9, 2.1, 1.8,
        2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2. , 1.9, 2.1, 2. , 2.4, 2.3, 1.8,
        2.2, 2.3, 1.5, 2.3, 2. , 2. , 1.8, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6,
        1.9, 2. , 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9,
        2.3, 2.5, 2.3, 1.9, 2. , 2.3, 1.8
        };
    std::vector<int> y_train = {
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
    sgdlib::FeatureScalarType inv_num_samples = 1.0 / static_cast<sgdlib::FeatureScalarType>(num_samples);
    sgdlib::ArrayDatasetType dataset(X_train, y_train, 150, 4);
    double y_hat;
    double loss = 0.0;
    int y;
    std::vector<double> x(4);

    std::vector<double> grad(num_features);
    for (std::size_t i = 0; i < num_samples; ++i) {
        dataset.X_row_data(i, x);
        dataset.y_row_data(i, y);
        y_hat = std::inner_product(x.begin(), x.end(),
                                   w.begin(), 0.0);

        loss += loss_fn->evaluate(y_hat, y);
        for (std::size_t j = 0; j < num_features; ++j) {
            grad[j] += loss_fn->derivate(y_hat, y) * x[j];
        }
    }
    loss /= static_cast<double>(num_samples);
    std::transform(grad.begin(), grad.end(), grad.begin(),
                  [num_samples](double val) {
                    return val / static_cast<double>(num_samples);
                });
    double tolerance = 1e-5;
    std::vector<double> expect_grad = {2.75991, 1.64394, 1.26731, 0.324662};

    EXPECT_NEAR(loss, 5.99602, tolerance);
    for (std::size_t i = 0; i < grad.size(); ++i) {
        EXPECT_NEAR(grad[i], expect_grad[i], tolerance);
    }

    // setup callback function
    std::vector<double> all_dlosses(num_samples);
    loss_fn->set_callback([&all_dlosses](const std::vector<double>& dloss_history) {
        all_dlosses.assign(dloss_history.begin(), dloss_history.end());
    });

    tolerance = 1e-5;
    std::vector<double> total_grad(num_features);
    // call evaluate_with_gradient to trigger callback function
    double total_loss = loss_fn->evaluate_with_gradient(dataset, w, total_grad);
    total_loss *= inv_num_samples;
    sgdlib::detail::vecscale<sgdlib::FeatureScalarType>(total_grad, inv_num_samples, total_grad);
    EXPECT_NEAR(total_loss, 5.99602, tolerance);
    for (std::size_t i = 0; i < grad.size(); ++i) {
        EXPECT_NEAR(total_grad[i], expect_grad[i], tolerance);
    }

    // std::cout << "all_dlosses size: " << all_dlosses.size() << std::endl;
    // for (auto dloss : all_dlosses) {
    //     std::cout << dloss << " ";
    // }
    // std::cout << std::endl;

}

