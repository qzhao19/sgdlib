#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/common/predefs.hpp"
#include "sgdlib/common/logging.hpp"

namespace sgdlib {

TEST(LoggingTest, PrintRuntimeInfoTest) {
    // define std::cout to stringstream
    std::stringstream buffer;
    std::streambuf* info = std::cout.rdbuf(buffer.rdbuf());

    // call PrintRuntimeInfo
    for (int i = 0; i < 10; i++) {
        print_runtime_info(3, "Current value: ", i, ", Status: OK");
    }

    // restore std::cout
    std::cout.rdbuf(info);

    // check
    std::string output = buffer.str();

    EXPECT_TRUE(output.find("Loop 3: Current value: 2, Status: OK") == std::string::npos);
    EXPECT_TRUE(output.find("Loop 6: Current value: 5, Status: OK") == std::string::npos);
    EXPECT_TRUE(output.find("Loop 9: Current value: 8, Status: OK") == std::string::npos);
}

TEST(LoggingTest, BasicExceptionThrow) {
    EXPECT_THROW({
        throw_error_msg<std::runtime_error>("Error code: ", 404, ", Message: Not Found");
    }, std::runtime_error);
}

TEST(LoggingTest, ValidErrorMessage) {
    try {
        throw_error_msg<std::invalid_argument>("Error code: ", 404, ", Message: Not Found");
        FAIL() << "Expected std::runtime_error, but no exception was thrown.";
    } catch (const std::invalid_argument& e) {
        // check that the exception is thrown
        EXPECT_STREQ(e.what(), "ERROR: Error code: 404, Message: Not Found\n");
    }
}

}
