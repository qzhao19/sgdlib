#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/common/registry.hpp"

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

class AnotherBar : public Foo {
public:
    explicit AnotherBar(const std::vector<int>& x) : Foo(x) {};

    virtual void Say() {
        std::cout <<  "I am AnotherBar" << std::endl;
    }

};
REGISTER_CLASS(FooRegistry, AnotherBar, AnotherBar);

TEST(TestRegistry, CreatorForFoo) {
    std::vector<int> x = {1};
    std::unique_ptr<Foo> bar = FooRegistry()->Create("Bar", x);
    bar->Say();
    EXPECT_TRUE(bar != nullptr) << "Cannot create bar";
    std::unique_ptr<Foo> another_bar = FooRegistry()->Create("AnotherBar", x);
    another_bar->Say();
    EXPECT_TRUE(another_bar != nullptr);
}


} // end of namespace


