#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/algorithm/base.hpp"
#include "sgdlib/algorithm/sgd/scd.hpp"

namespace sgdlib {

class SCDTest : public ::testing::Test {
public:
    virtual void SetUp() {
        std::vector<FeatValType> w0 = {1.0, 1.0, 1.0, 1.0}; 
        std::string loss = "LogLoss";
        double alpha = 0.01;
        double tol = 0.0001;
        std::size_t max_iters = 1000; 
        std::size_t random_seed = -1;
        bool shuffle = true;
        bool verbose = true;

        optimizer = std::make_unique<sgdlib::SCD>(w0,
            loss, 
            alpha, 
            tol, 
            max_iters, 
            random_seed,
            shuffle, 
            verbose
        );
    }

    std::unique_ptr<sgdlib::BaseOptimizer> optimizer;
};


TEST_F(SCDTest, SCDOptimizerTest) {
    std::vector<double> X_train = {
        -0.5557,0.25,-0.8643,-0.9165,-0.6665,-0.1666,-0.8643,-0.9165,-0.778,0.,-0.8984,-0.9165,-0.8335,-0.0833,-0.8306,-0.9165,-0.6113,
        0.3333,-0.8643,-0.9165,-0.389,0.5835,-0.7627,-0.75,-0.8335,0.1666,-0.8643,-0.8335,-0.6113,0.1666,-0.8306,-0.9165,-0.9443,-0.25,
        -0.8643,-0.9165,-0.6665,-0.0833,-0.8306,-1.,-0.389,0.4167,-0.8306,-0.9165,-0.722,0.1666,-0.7964,-0.9165,-0.722,-0.1666,-0.8643,
        -1.,-1.,-0.1666,-0.9663,-1.,-0.1666,0.6665,-0.932,-0.9165,-0.2222,1.,-0.8306,-0.75,-0.389,0.5835,-0.8984,-0.75,-0.5557,0.25,-0.8643,
        -0.8335,-0.2222,0.5,-0.7627,-0.8335,-0.5557,0.5,-0.8306,-0.8335,-0.389,0.1666,-0.7627,-0.9165,-0.5557,0.4167,-0.8306,-0.75,-0.8335,
        0.3333,-1.,-0.9165,-0.5557,0.0833,-0.7627,-0.6665,-0.722,0.1666,-0.695,-0.9165,-0.6113,-0.1666,-0.7964,-0.9165,-0.6113,0.1666,-0.7964,
        -0.75,-0.5,0.25,-0.8306,-0.9165,-0.5,0.1666,-0.8643,-0.9165,-0.778,0.,-0.7964,-0.9165,-0.722,-0.0833,-0.7964,-0.9165,-0.389,0.1666,
        -0.8306,-0.75,-0.5,0.75,-0.8306,-1.,-0.3333,0.8335,-0.8643,-0.9165,-0.6665,-0.0833,-0.8306,-0.9165,-0.6113,0.,-0.932,-0.9165,-0.3333,
        0.25,-0.8984,-0.9165,-0.6665,0.3333,-0.8643,-1.,-0.9443,-0.1666,-0.8984,-0.9165,-0.5557,0.1666,-0.8306,-0.9165,-0.6113,0.25,-0.8984,
        -0.8335,-0.8887,-0.75,-0.8984,-0.8335,-0.9443,0.,-0.8984,-0.9165,-0.6113,0.25,-0.7964,-0.5835,-0.5557,0.5,-0.695,-0.75,-0.722,-0.1666,
        -0.8643,-0.8335,-0.5557,0.5,-0.7964,-0.9165,-0.8335,0.,-0.8643,-0.9165,-0.4443,0.4167,-0.8306,-0.9165,-0.6113,0.0833,-0.8643,-0.9165,
        0.5,0.,0.2542,0.0833,0.1666,0.,0.1864,0.1666,0.4443,-0.0833,0.322,0.1666,-0.3333,-0.75,0.01695,0.,0.2222,-0.3333,0.2203,0.1666,-0.2222,
        -0.3333,0.1864,0.,0.1111,0.0833,0.2542,0.25,-0.6665,-0.6665,-0.2203,-0.25,0.2778,-0.25,0.2203,0.,-0.5,-0.4167,-0.01695,0.0833,-0.6113,-1.,
        -0.1526,-0.25,-0.1111,-0.1666,0.0847,0.1666,-0.05554,-0.8335,0.01695,-0.25,-0.,-0.25,0.2542,0.0833,-0.2778,-0.25,-0.11865,0.,0.3333,
        -0.0833,0.1526,0.0833,-0.2778,-0.1666,0.1864,0.1666,-0.1666,-0.4167,0.05084,-0.25,0.05554,-0.8335,0.1864,0.1666,-0.2778,-0.5835,-0.01695,
        -0.1666,-0.1111,0.,0.288,0.4167,-0.,-0.3333,0.01695,0.,0.1111,-0.5835,0.322,0.1666,-0.,-0.3333,0.2542,-0.0833,0.1666,-0.25,0.11865,0.,0.2778,
        -0.1666,0.1526,0.0833,0.389,-0.3333,0.288,0.0833,0.3333,-0.1666,0.356,0.3333,-0.05554,-0.25,0.1864,0.1666,-0.2222,-0.5,-0.1526,-0.25,-0.3333,
        -0.6665,-0.05084,-0.1666,-0.3333,-0.6665,-0.0847,-0.25,-0.1666,-0.4167,-0.01695,-0.0833,-0.05554,-0.4167,0.39,0.25,-0.389,-0.1666,0.1864,
        0.1666,-0.05554,0.1666,0.1864,0.25,0.3333,-0.0833,0.2542,0.1666,0.1111,-0.75,0.1526,0.,-0.2778,-0.1666,0.05084,0.,-0.3333,-0.5835,0.01695,
        0.,-0.3333,-0.5,0.1526,-0.0833,-0.,-0.1666,0.2203,0.0833,-0.1666,-0.5,0.01695,-0.0833,-0.6113,-0.75,-0.2203,-0.25,-0.2778,-0.4167,0.0847,0.,
        -0.2222,-0.1666,0.0847,-0.0833,-0.2222,-0.25,0.0847,0.,0.05554,-0.25,0.11865,0.,-0.5557,-0.5835,-0.322,-0.1666,-0.2222,-0.3333,0.05084,0.,
        0.1111,0.0833,0.695,1.,-0.1666,-0.4167,0.39,0.5,0.5557,-0.1666,0.661,0.6665,0.1111,-0.25,0.559,0.4167,0.2222,-0.1666,0.627,0.75,0.8335,
        -0.1666,0.8984,0.6665,-0.6665,-0.5835,0.1864,0.3333,0.6665,-0.25,0.7964,0.4167,0.3333,-0.5835,0.627,0.4167,0.6113,0.3333,0.729,1.,0.2222,
        0.,0.39,0.5835,0.1666,-0.4167,0.4575,0.5,0.389,-0.1666,0.5254,0.6665,-0.2222,-0.5835,0.356,0.5835,-0.1666,-0.3333,0.39,0.9165,0.1666,0.,
        0.4575,0.8335,0.2222,-0.1666,0.5254,0.4167,0.8887,0.5,0.932,0.75,0.8887,-0.5,1.,0.8335,-0.05554,-0.8335,0.356,0.1666,0.4443,0.,0.5933,
        0.8335,-0.2778,-0.3333,0.322,0.5835,0.8887,-0.3333,0.932,0.5835,0.1111,-0.4167,0.322,0.4167,0.3333,0.0833,0.5933,0.6665,0.6113,0.,0.695,
        0.4167,0.05554,-0.3333,0.288,0.4167,-0.,-0.1666,0.322,0.4167,0.1666,-0.3333,0.559,0.6665,0.6113,-0.1666,0.627,0.25,0.722,-0.3333,0.729,
        0.5,1.,0.5,0.8306,0.5835,0.1666,-0.3333,0.559,0.75,0.1111,-0.3333,0.39,0.1666,-0.,-0.5,0.559,0.0833,0.8887,-0.1666,0.729,0.8335,0.1111,
        0.1666,0.559,0.9165,0.1666,-0.0833,0.5254,0.4167,-0.05554,-0.1666,0.288,0.4167,0.4443,-0.0833,0.4915,0.6665,0.3333,-0.0833,0.559,0.9165,
        0.4443,-0.0833,0.39,0.8335,-0.1666,-0.4167,0.39,0.5,0.389,0.,0.661,0.8335,0.3333,0.0833,0.5933,1.,0.3333,-0.1666,0.4238,0.8335,0.1111,
        -0.5835,0.356,0.5,0.2222,-0.1666,0.4238,0.5835,0.05554,0.1666,0.4915,0.8335,-0.1111,-0.1666,0.39,0.4167,};
    std::vector<long> y_train = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    optimizer->optimize(X_train, y_train);

    std::vector<double> coef;
    double intercept;

    coef = optimizer->get_coef();
    // intercept = optimizer->get_intercept();

    std::cout << "coefficients = ";
    for (auto c : coef) {
        std::cout << c << " ";
    }
    std::cout << std::endl;
    // std::cout << "intercept = " << intercept <<std::endl;
};
    
}




