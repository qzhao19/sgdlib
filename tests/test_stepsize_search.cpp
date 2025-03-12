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
        std::vector<double> X = {5.2, 3.3, 1.2, 0.3,
                                4.8, 3.1 , 1.6, 0.2,
                                4.75, 3.1, 1.32, 0.1,
                                5.9, 2.6, 4.1, 1.2,
                                5.1, 2.2, 3.3, 1.1,
                                5.2, 2.7, 4.1, 1.3,
                                6.6, 3.1, 5.25, 2.2,
                                6.3, 2.5, 5.1, 2.0,
                                6.5, 3.1, 5.2, 2.1};
        std::vector<long> y = {-1, -1, -1, -1, 1, 1, 1, 1, 1};
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
        else if (search_policy == "BacktrackingLineSearch") {
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
    EXPECT_DOUBLE_EQ(stepsize, 0.046204048629761185);
}

TEST_F(StepSizeSearchTest, BasicLineSearchTest) {
    SetUp("BasicLineSearch");
    FeatValType y_pred = 2.95782, y_true = 1, dloss = -0.049368, xnorm = 98.63;
    std::size_t index = 120;
    double stepsize = 0.0;
    stepsize_search_->search(y_pred, y_true, dloss, xnorm, index, stepsize);
    EXPECT_DOUBLE_EQ(stepsize, 0.021601194777846125);
}


TEST_F(StepSizeSearchTest, BacktrackingLineSearchTest) {
    SetUp("BacktrackingLineSearch");
    std::vector<double> x = {1., 1., 1., 1.}; 
    double fx = 5.996018216462658; 
    std::vector<double> g = {2.75991281, 1.64394249, 1.26730677, 0.32466222};
    std::vector<double> gp = g;
    std::vector<double> d = {-2.75991281, -1.64394249, -1.26730677, -0.32466222};
    double stepsize = 0.2883013346297236;
    std::vector<double> xp = {1., 1., 1., 1.};
    stepsize_search_->search(xp, gp, d, x, g, fx, stepsize);
    double tolerance = 1e-5;
    EXPECT_NEAR(stepsize, 0.605433, tolerance);
    EXPECT_NEAR(fx,  1.2725, tolerance);
    std::vector<double> expect1 = {-0.670942, 0.00470329, 0.232731, 0.803439};
    std::vector<double> expect2 = {-2.23968, -0.972497, -1.88325, -0.741734};
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(expect1[i], x[i], tolerance);
        EXPECT_NEAR(expect2[i], g[i], tolerance);
    }  
}

}

