#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/common/registry.hpp"

namespace sgdlib {

class Foo {
public:
    explicit Foo(int x) { 
        std::cout << "Foo" <<std::endl; 
    }
};

DECLARE_REGISTRY(FooRegistry, Foo, int);
DEFINE_REGISTRY(FooRegistry, Foo, int);

class Bar : public Foo {
public:
    explicit Bar(int x) : Foo(x) { 
        std::cout << "Bar" <<std::endl;  
    }
};
REGISTER_CLASS(FooRegistry, Bar, Bar);

class AnotherBar : public Foo {
public:
    explicit AnotherBar(int x) : Foo(x) { 
        std::cout << "Another Bar" <<std::endl;  
    }
};
REGISTER_CLASS(FooRegistry, AnotherBar, AnotherBar);

TEST(TestRegistry, CanRunCreator) {
    std::unique_ptr<Foo> bar(FooRegistry()->Create("Bar", 1));
    EXPECT_TRUE(bar != nullptr) << "Cannot create bar";
    std::unique_ptr<Foo> another_bar(FooRegistry()->Create("AnotherBar", 1));
    EXPECT_TRUE(another_bar != nullptr);
}

} // end of namespace


