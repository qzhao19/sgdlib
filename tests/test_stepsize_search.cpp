#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/core/loss.hpp"
#include "sgdlib/core/stepsize_search.hpp"

namespace sgdlib {

class StepSizeSearchTest : public ::testing::Test {
public:
    virtual void SetUp(std::string search_policy) {
        std::vector<double> X = {
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
        std::vector<long> y = {
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        LossParamType loss_params = {{"alpha", 0.0}};
        StepSizeSearchParamType* stepsize_search_params = &DEFAULT_STEPSIZE_SEARCH_PARAMS;
        loss_fn_ = LossFunctionRegistry()->Create("LogLoss", loss_params);
        if (search_policy == "Constant") {
            stepsize_search_ = std::make_unique<sgdlib::ConstantSearch<sgdlib::LossFunction>>(
                X, y, loss_fn_, stepsize_search_params
            );
        }
        else if (search_policy == "BasicLineSearch") {
            stepsize_search_params->alpha = 0.0;
            stepsize_search_params->eta0 = 0.01;
            stepsize_search_params->max_searches = 10;
            stepsize_search_params->max_iters = 20;
            stepsize_search_ = std::make_unique<sgdlib::BasicLineSearch<sgdlib::LossFunction>>(
                X, y, loss_fn_, stepsize_search_params
            );
        }
        else if (search_policy == "BacktrackingLineSearch" || search_policy == "BracketingLineSearch") {
            stepsize_search_params->max_searches = 40;
            stepsize_search_params->condition = "WOLFE";
            stepsize_search_ = std::make_unique<sgdlib::BacktrackingLineSearch<sgdlib::LossFunction>>(
                X, y, loss_fn_, stepsize_search_params
            );
        }
    }
    
    std::shared_ptr<sgdlib::LossFunction> loss_fn_; 
    std::unique_ptr<sgdlib::StepSizeSearch<sgdlib::LossFunction>> stepsize_search_;
};

TEST_F(StepSizeSearchTest, ConstantSearchTest) {
    SetUp("Constant");
    double stepsize = 0.0;
    stepsize_search_->search(false, stepsize);
    EXPECT_DOUBLE_EQ(stepsize, 0.03213883978788365);
}

TEST_F(StepSizeSearchTest, BasicLineSearchTest) {
    SetUp("BasicLineSearch");
    FeatValType y_pred = 2.95782, y_true = 1, dloss = -0.049368, xnorm = 98.63;
    std::size_t index = 120;
    double stepsize = 0.0;
    stepsize_search_->search(y_pred, y_true, dloss, xnorm, index, stepsize);
    EXPECT_DOUBLE_EQ(stepsize, 0.010472941228206268);
}

TEST_F(StepSizeSearchTest, BacktrackingLineSearchTest) {
    SetUp("BacktrackingLineSearch");
    double tolerance = 1e-5;
    std::vector<std::vector<double>> x = {{1., 1., 1., 1.}, 
                                           {-0.67094175, 0.00470329, 0.23273091, 0.80343884}};
    std::vector<double> fx = {5.996018216462658, 0.9177489446446626};
    std::vector<std::vector<double>> g = {{2.75991281, 1.64394249, 1.26730677, 0.32466222}, 
                                           {-2.32159527, -1.03857245, -1.92900394, -0.68107431}};
    std::vector<std::vector<double>> d {{-2.75991281, -1.64394249, -1.26730677, -0.32466222}, 
                                         {0.7449502,  0.38705061, 0.48413709, 0.154796}};
    std::vector<std::vector<double>> xp = x;
    double stepsize = 0.2883013346297236;

    std::vector<std::vector<double>> expect_x = {{-0.67094175,  0.00470329,  0.23273091,  0.80343884}, 
                                                 {-0.21992446,  0.23903643,  0.52584339,  0.89715742}};
    std::vector<std::vector<double>> expect_g = {{-2.32159528, -1.03857245, -1.92900395, -0.68107431}, 
                                                 {2.04742451, 1.19381727, 0.98833884, 0.26008274}};
    std::vector<double> expect_fx = {0.9177489530218861, 0.9059209513061773};
    std::vector<double> expect_stepsize = {0.6054328027224196, 0.6054328027224196};

    for (std::size_t i = 0; i < 2; ++i) {
        int status = stepsize_search_->search(xp[i], g[i], d[i], x[i], g[i], fx[i], stepsize);

        EXPECT_NEAR(stepsize, expect_stepsize[i], tolerance);
        EXPECT_NEAR(fx[i], expect_fx[i], tolerance);

        for (std::size_t j = 0; j < x[i].size(); ++j) {
            EXPECT_NEAR(x[i][j], expect_x[i][j], tolerance);
            EXPECT_NEAR(g[i][j], expect_g[i][j], tolerance);
        }  
    }
}

TEST_F(StepSizeSearchTest, BracketingLineSearchTest) {
    SetUp("BracketingLineSearch");
    double tolerance = 1e-5;
    std::vector<std::vector<double>> x = {{1., 1., 1., 1.}, 
                                           {-0.67094175, 0.00470329, 0.23273091, 0.80343884}};
    std::vector<double> fx = {5.996018216462658, 0.9177489446446626};
    std::vector<std::vector<double>> g = {{2.75991281, 1.64394249, 1.26730677, 0.32466222}, 
                                           {-2.32159527, -1.03857245, -1.92900394, -0.68107431}};
    std::vector<std::vector<double>> d {{-2.75991281, -1.64394249, -1.26730677, -0.32466222}, 
                                         {0.7449502,  0.38705061, 0.48413709, 0.154796}};
    std::vector<std::vector<double>> xp = x;
    double stepsize = 0.2883013346297236;

    std::vector<std::vector<double>> expect_x = {{-0.67094175,  0.00470329,  0.23273091,  0.80343884}, 
                                                 {-0.21992446,  0.23903643,  0.52584339,  0.89715742}};
    std::vector<std::vector<double>> expect_g = {{-2.32159528, -1.03857245, -1.92900395, -0.68107431}, 
                                                 {2.04742451, 1.19381727, 0.98833884, 0.26008274}};
    std::vector<double> expect_fx = {0.9177489530218861, 0.9059209513061773};
    std::vector<double> expect_stepsize = {0.6054328027224196, 0.6054328027224196};

    for (std::size_t i = 0; i < 2; ++i) {
        int status = stepsize_search_->search(xp[i], g[i], d[i], x[i], g[i], fx[i], stepsize);

        EXPECT_NEAR(stepsize, expect_stepsize[i], tolerance);
        EXPECT_NEAR(fx[i], expect_fx[i], tolerance);

        for (std::size_t j = 0; j < x[i].size(); ++j) {
            EXPECT_NEAR(x[i][j], expect_x[i][j], tolerance);
            EXPECT_NEAR(g[i][j], expect_g[i][j], tolerance);
        }  
    }
}


}

