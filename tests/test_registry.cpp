#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/common/registry.hpp"

namespace sgdlib {

class Foo {
public:
    explicit Foo(int x) {}
    virtual void Say() = 0;
};

DECLARE_REGISTRY(FooRegistry, Foo, int);
DEFINE_REGISTRY(FooRegistry, Foo, int);

class Bar : public Foo {
public:
    explicit Bar(int x) : Foo(x) {}

    virtual void Say() {
        std::cout << "I am Bar" << std::endl;
    }
};
REGISTER_CLASS(FooRegistry, Bar, Bar);

class AnotherBar : public Foo {
public:
    explicit AnotherBar(int x) : Foo(x) { }

    virtual void Say() {
        std::cout << "I am Another Bar" << std::endl;
    }
};
REGISTER_CLASS(FooRegistry, AnotherBar, AnotherBar);

TEST(TestRegistry, CanRunCreator) {
    std::unique_ptr<Foo> bar = FooRegistry()->Create("Bar", 1);
    bar->Say();

    EXPECT_TRUE(bar != nullptr) << "Cannot create bar";
    std::unique_ptr<Foo> another_bar(FooRegistry()->Create("AnotherBar", 1));
    EXPECT_TRUE(another_bar != nullptr);
}

} // end of namespace


