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
        stepsize_search_params->alpha = 0.0;
        stepsize_search_params->eta0 = 0.01;
        stepsize_search_params->max_searches = 10;
        stepsize_search_params->max_iters = 20;

        loss_fn_ = LossFunctionRegistry()->Create("LogLoss", loss_params);
        if (search_policy == "Constant") {
            stepsize_search_ = std::make_unique<sgdlib::ConstantSearch<sgdlib::LossFunction>>(
                X, y, loss_fn_, stepsize_search_params
            );
        }
        else if (search_policy == "BasicLineSearch") {
            stepsize_search_ = std::make_unique<sgdlib::BasicLineSearch<sgdlib::LossFunction>>(
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
    FeatureType y_pred = 2.95782, y_true = 1, dloss = -0.049368, xnorm = 98.63;
    std::size_t index = 120;
    double stepsize = 0.0;
    stepsize_search_->search(y_pred, y_true, dloss, xnorm, index, stepsize);
    EXPECT_DOUBLE_EQ(stepsize, 0.021601194777846125);
}


}

