#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

// include private header file
#include "sgdlib/common/registry.hpp"

namespace sgdlib {
namespace detail {

class Foo {
public:
    Foo(const std::vector<int>& x) {};
    virtual ~Foo() = default;
    virtual void Say() = 0;
};

DECLARE_SHARED_REGISTRY(FooRegistry, Foo, std::vector<int>);
DEFINE_SHARED_REGISTRY(FooRegistry, Foo, std::vector<int>);

class Bar : public Foo {
public:
    Bar(const std::vector<int>& x) : Foo(x) {};

    void Say() override {
        std::cout <<  "I am Bar" << std::endl;
    }
};
REGISTER_CLASS(FooRegistry, Bar, Bar);

class AnotherBar : public Foo {
public:
    explicit AnotherBar(const std::vector<int>& x) : Foo(x) {};

    void Say() override {
        std::cout <<  "I am AnotherBar" << std::endl;
    }

};
REGISTER_CLASS(FooRegistry, AnotherBar, AnotherBar);

} // namespace detail
} // namespace sgdlib


TEST(TestRegistry, CreatorForFoo) {
    std::vector<int> x = {1};
    std::shared_ptr<sgdlib::detail::Foo> bar = sgdlib::detail::FooRegistry()->Create("Bar", x);
    bar->Say();
    EXPECT_TRUE(bar != nullptr) << "Cannot create bar";
    std::shared_ptr<sgdlib::detail::Foo> another_bar = sgdlib::detail::FooRegistry()->Create("AnotherBar", x);
    another_bar->Say();
    EXPECT_TRUE(another_bar != nullptr);
}
