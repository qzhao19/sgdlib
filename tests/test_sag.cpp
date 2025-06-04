#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/algorithm/base.hpp"
#include "sgdlib/algorithm/sgd/sag.hpp"

class SAGTest : public ::testing::Test {
public:
    void SetUp(bool is_saga) {
        X_train = {
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
        y_train = {
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        
        w0 = {1.0, 1.0, 1.0, 1.0}; 
        b0 = 1.0;
        std::string loss = "LogLoss";
        std::string search_policy = "BasicLineSearch";
        double alpha = 0.0;
        double eta0 = 0.01;
        double tol = 0.005;
        double gamma = 0.5;
        std::size_t max_iters = 100; 
        std::size_t num_iters_no_change = 5;
        std::size_t random_seed = 1;
        // bool is_saga = false;
        bool shuffle = true;
        bool verbose = false;

        optimizer = std::make_unique<sgdlib::SAG>(w0, b0,
            loss, 
            search_policy,
            alpha,
            eta0,
            tol, 
            max_iters, 
            random_seed,
            is_saga,
            shuffle, 
            verbose
        );
    }
    std::vector<double> X_train;
    std::vector<long> y_train;
    std::vector<double> w0;
    double b0;
    std::unique_ptr<sgdlib::BaseOptimizer> optimizer;
};


TEST_F(SAGTest, SAGBasicOptimizationTest) {
    bool is_saga = false;
    SetUp(is_saga);
    optimizer->optimize(X_train, y_train);
    
    const auto& w_opt = optimizer->get_weights();
    const auto b_opt = optimizer->get_intercept();
    
    // check weight update
    EXPECT_EQ(w_opt.size(), 4);
    EXPECT_FALSE(w_opt[0] == w0[0]) << "Weights not updated";
    EXPECT_FALSE(w_opt[1] == w0[1]) << "Weights not updated";
    
    // check bias update
    EXPECT_FALSE(b_opt == b0) << "Bias not updated";

    // print 
    std::cout << "coefficients = ";
    for (auto w : w_opt) {
        std::cout << w << " ";
    }
    std::cout << std::endl;
    std::cout << "intercept = " << b_opt <<std::endl;
}

TEST_F(SAGTest, SAGConvergenceTest) {
    bool is_saga = false;
    SetUp(is_saga);
    EXPECT_NO_THROW(optimizer->optimize(X_train, y_train));
    EXPECT_TRUE(optimizer->get_weights().size() > 0);
}

TEST_F(SAGTest, SAGConvergenceSpeedTest) {
    bool is_saga = false;
    SetUp(is_saga);
    
    std::vector<double> all_losses;
    all_losses.reserve(100 * 150);
    optimizer->set_callback([&all_losses](const std::vector<double>& loss_history) {
        all_losses.insert(all_losses.end(), loss_history.begin(), loss_history.end());
    });
    optimizer->optimize(X_train, y_train);
    all_losses.shrink_to_fit();
    // std::cout << "losses size = " << all_losses.size() << std::endl;
    
    const double initial_loss = all_losses[0];
    const double final_loss = all_losses.back();
    const double improvement_ratio = (initial_loss - final_loss) / initial_loss;

    EXPECT_GT(improvement_ratio, 0.3f) << "insufficient convergence rate";
};


TEST_F(SAGTest, SAGABasicOptimizationTest) {
    bool is_saga = true;
    SetUp(is_saga);
    optimizer->optimize(X_train, y_train);
    
    const auto& w_opt = optimizer->get_weights();
    const auto b_opt = optimizer->get_intercept();
    
    // check weight update
    EXPECT_EQ(w_opt.size(), 4);
    EXPECT_FALSE(w_opt[0] == w0[0]) << "Weights not updated";
    EXPECT_FALSE(w_opt[1] == w0[1]) << "Weights not updated";
    
    // check bias update
    EXPECT_FALSE(b_opt == b0) << "Bias not updated";

    // print 
    std::cout << "coefficients = ";
    for (auto w : w_opt) {
        std::cout << w << " ";
    }
    std::cout << std::endl;
    std::cout << "intercept = " << b_opt <<std::endl;
}

TEST_F(SAGTest, SAGAConvergenceTest) {
    bool is_saga = true;
    SetUp(is_saga);
    EXPECT_NO_THROW(optimizer->optimize(X_train, y_train));
    EXPECT_TRUE(optimizer->get_weights().size() > 0);
}

TEST_F(SAGTest, SAGAConvergenceSpeedTest) {
    bool is_saga = true;
    SetUp(is_saga);
    
    std::vector<double> all_losses;
    all_losses.reserve(100 * 150);
    optimizer->set_callback([&all_losses](const std::vector<double>& loss_history) {
        all_losses.insert(all_losses.end(), loss_history.begin(), loss_history.end());
    });
    optimizer->optimize(X_train, y_train);
    all_losses.shrink_to_fit();
    // std::cout << "losses size = " << all_losses.size() << std::endl;

    const double initial_loss = all_losses[0];
    const double final_loss = all_losses.back();
    const double improvement_ratio = (initial_loss - final_loss) / initial_loss;

    EXPECT_GT(improvement_ratio, 0.3f) << "insufficient convergence rate";
};
