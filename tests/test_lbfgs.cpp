#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/algorithm/base.hpp"
#include "sgdlib/algorithm/lbfgs/lbfgs.hpp"

class LBFGSTest : public ::testing::Test {
public:
    virtual void SetUp(std::string search_policy) {
        X_train = {
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
        y_train = {
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        };

        w0 = {1.0, 1.0, 1.0, 1.0};
        std::string loss = "LogLoss";
        // std::string search_policy = "BacktrackingLineSearch";
        double delta = 1e-6;
        double tol = 1e-5;
        std::size_t max_iters = 0;
        std::size_t mem_size = 8;
        std::size_t past = 3;
        bool verbose = false;
        dataset = std::make_unique<sgdlib::ArrayDatasetType>(
            X_train, y_train, 150, 4
        );

        std::shared_ptr<sgdlib::StepSizeSearchParamType> stepsize_search_params = std::make_shared<sgdlib::StepSizeSearchParamType>(
            sgdlib::detail::DEFAULT_STEPSIZE_SEARCH_PARAMS
        );
        stepsize_search_params->max_searches = 40;
        stepsize_search_params->max_iters = 20;
        optimizer = std::make_unique<sgdlib::LBFGS>(w0,
            loss,
            search_policy,
            delta,
            tol,
            max_iters,
            mem_size,
            past,
            stepsize_search_params,
            verbose
        );
    }
    std::vector<double> X_train;
    std::vector<int> y_train;
    std::vector<double> w0;
    std::unique_ptr<sgdlib::Optimizer> optimizer;
    std::unique_ptr<sgdlib::ArrayDatasetType> dataset;
};

TEST_F(LBFGSTest, LBFGSWithBacktrackingTest) {
    SetUp("BacktrackingLineSearch");

    optimizer->optimize(*dataset);

    std::vector<double> coef;
    double intercept;

    coef = optimizer->get_weights();
    // intercept = optimizer->get_intercept();

    std::cout << "coefficients = ";
    for (auto c : coef) {
        std::cout << c << " ";
    }
    std::cout << std::endl;
};

TEST_F(LBFGSTest, LBFGSWithBacktrackingConvergenceTest) {
    SetUp("BacktrackingLineSearch");
    EXPECT_NO_THROW(optimizer->optimize(*dataset));
    EXPECT_TRUE(optimizer->get_weights().size() > 0);
}

TEST_F(LBFGSTest, LBFGSWithBacktrackingConvergenceSpeedTest) {
    SetUp("BacktrackingLineSearch");
    std::vector<double> all_losses;
    all_losses.reserve(100 * 150);
    optimizer->set_callback([&all_losses](const std::vector<double>& loss_history) {
        all_losses.insert(all_losses.end(), loss_history.begin(), loss_history.end());
    });
    optimizer->optimize(*dataset);
    all_losses.shrink_to_fit();
    // std::cout << "losses size = " << all_losses.size() << std::endl;

    const double initial_loss = all_losses[0];
    const double final_loss = all_losses.back();
    const double improvement_ratio = (initial_loss - final_loss) / initial_loss;

    EXPECT_GT(improvement_ratio, 0.3f) << "insufficient convergence rate";
};


TEST_F(LBFGSTest, LBFGSWithBracketingTest) {
    SetUp("BracketingLineSearch");

    optimizer->optimize(*dataset);

    std::vector<double> coef;
    double intercept;
    coef = optimizer->get_weights();

    std::cout << "coefficients = ";
    for (auto c : coef) {
        std::cout << c << " ";
    }
    std::cout << std::endl;
};

TEST_F(LBFGSTest, LBFGSWithBracketingConvergenceTest) {
    SetUp("BracketingLineSearch");
    EXPECT_NO_THROW(optimizer->optimize(*dataset));
    EXPECT_TRUE(optimizer->get_weights().size() > 0);
}

TEST_F(LBFGSTest, LBFGSWithBracketingConvergenceSpeedTest) {
    SetUp("BracketingLineSearch");
    std::vector<double> all_losses;
    all_losses.reserve(100 * 150);
    optimizer->set_callback([&all_losses](const std::vector<double>& loss_history) {
        all_losses.insert(all_losses.end(), loss_history.begin(), loss_history.end());
    });
    optimizer->optimize(*dataset);
    all_losses.shrink_to_fit();
    // std::cout << "losses size = " << all_losses.size() << std::endl;

    const double initial_loss = all_losses[0];
    const double final_loss = all_losses.back();
    const double improvement_ratio = (initial_loss - final_loss) / initial_loss;

    EXPECT_GT(improvement_ratio, 0.3f) << "insufficient convergence rate";
};
