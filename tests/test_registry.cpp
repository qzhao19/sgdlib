#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/common/registry.hpp"
#include "sgdlib/core/loss/log_loss.hpp"

namespace sgdlib {

class Foo {
public:
    Foo(const std::vector<int>& x) {};
    virtual void Say() = 0;
};

DECLARE_REGISTRY(FooRegistry, Foo, std::vector<int>);
DEFINE_REGISTRY(FooRegistry, Foo, std::vector<int>);

class Bar : public Foo {
public:
    Bar(const std::vector<int>& x) : Foo(x) {};

    virtual void Say() {
        std::cout <<  "I am Bar" << std::endl;
    }
};
REGISTER_CLASS(FooRegistry, Bar, Bar);

TEST(TestRegistry, CanRunCreator) {
    std::vector<int> x = {1};
    std::unique_ptr<Foo> bar = FooRegistry()->Create("Bar", x);
    bar->Say();
    EXPECT_TRUE(bar != nullptr) << "Cannot create bar";
}


TEST(TestRegistry, CanRunCreatorForLogLoss) {
    std::vector<double> X = {5.2, 3.3, 1.2, 0.3,
                            4.8, 3.1 , 1.6, 0.2,
                            4.75, 3.1, 1.32, 0.1,
                            5.9, 2.6, 4.1, 1.2,
                            5.1, 2.2, 3.3, 1.1,
                            5.2, 2.7, 4.1, 1.3,
                            6.6, 3.1, 5.25, 2.2,
                            6.3, 2.5, 5.1, 2.0,
                            6.5, 3.1, 5.2, 2.1};
    std::vector<long> y = {0, 0, 0, 0, 1, 1, 1, 1, 1};
    std::vector<double> w = {0.9781, 0.9711, 0.3962, 0.5209};

    LossParamType loss_param = {{"alpha", 0.0}};
    std::unique_ptr<sgdlib::LossFunction> loss_fn = LossFunctionRegistry()->Create("LogLoss", loss_param);
    double loss = loss_fn->evaluate(X, y, w);
    std::cout << loss << std::endl;

    EXPECT_DOUBLE_EQ(loss, 4.0159203584860403);

    std::vector<double> grad(4);
    loss_fn->gradient(X, y, w, grad);

    std::vector<double> expect ={2.2939883890082928, 1.3441739235904036, 
                                 0.9131519457304953, 0.19995927206507597};
    EXPECT_EQ(grad, expect);
    
}


} // end of namespace


