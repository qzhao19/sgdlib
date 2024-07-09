#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/core/stepsize_search.hpp"

namespace sgdlib {

class StepSizeSearchTest : public ::testing::Test {
public:
    virtual void SetUp() {
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
        StepSizeSearchParamType stepsize_search_param;
        stepsize_search_param["alpha"] = 0.0;

        stepsize_search_ = std::make_unique<sgdlib::ConstantSearch>(X, y, stepsize_search_param);
    }
    std::unique_ptr<sgdlib::StepSizeSearch> stepsize_search_;
};

TEST_F(StepSizeSearchTest, ConstantSearchTest) {
    double stepsize = 0.0;
    stepsize_search_->search(false, stepsize);
    EXPECT_DOUBLE_EQ(stepsize, 0.39022332435473611);
}


}

