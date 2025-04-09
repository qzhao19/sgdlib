#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "sgdlib/common/logging.hpp"

TEST(LoggingTest, PrintRuntimeInfoTest) {
    // define std::cout to stringstream
    std::stringstream buffer;
    std::streambuf* info = std::cout.rdbuf(buffer.rdbuf());
    // call PrintRuntimeInfo
    for (int i = 0; i < 10; i++) {
        sgdlib::detail::print_runtime_info(3, "Current value: ", i, ", Status: OK");
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
        sgdlib::detail::throw_error_msg<std::runtime_error>("Error code: ", 404, ", Message: Not Found");
    }, std::runtime_error);
}

TEST(LoggingTest, ValidErrorMessage) {
    try {
        sgdlib::detail::throw_error_msg<std::invalid_argument>("Error code: ", 404, ", Message: Not Found");
        FAIL() << "Expected std::runtime_error, but no exception was thrown.";
    } catch (const std::invalid_argument& e) {
        // check that the exception is thrown
        EXPECT_STREQ(e.what(), "ERROR: Error code: 404, Message: Not Found\n");
    }
}

class LoggerTest : public ::testing::Test {
public:
    void SetUp() override {
        // temp log path
        temp_log_dir_ = "./logs";
        std::filesystem::create_directories(temp_log_dir_);

        was_initialized_ = sgdlib::detail::Logger::is_initialized();

        // make sure that testing env is clean
        if (sgdlib::detail::Logger::is_initialized()) {
            google::ShutdownGoogleLogging();
        }
    }

    void TearDown() override {
        // clean logger
        if (!was_initialized_ && sgdlib::detail::Logger::is_initialized()) {
            // google::ShutdownGoogleLogging();
            std::cout << "ShutdownGoogleLogging" << std::endl;
        }

        // remove temp dir
        std::filesystem::remove_all(temp_log_dir_);
    }
    bool was_initialized_;
    std::filesystem::path temp_log_dir_;
};

TEST_F(LoggerTest, InitialStateIsNotInitialized) {
    EXPECT_FALSE(sgdlib::detail::Logger::is_initialized());
}


TEST_F(LoggerTest, SuccessfulInitializationWithMarco) {
    // make sure not init before call is_initialized
    ASSERT_FALSE(sgdlib::detail::Logger::is_initialized());

    // init Logger
    // EXPECT_NO_THROW({
    //     sgdlib::detail::Logger::initialize("test", temp_log_dir_.string());
    // });
    SGDLIB_LOG(INFO) << "This is a test message from SGDLIB_LOG macro";

    // check initialized status
    EXPECT_TRUE(sgdlib::detail::Logger::is_initialized());

    // check if temp_log_dir_ exist
    EXPECT_TRUE(std::filesystem::exists(temp_log_dir_));
    EXPECT_TRUE(std::filesystem::is_directory(temp_log_dir_));

    bool log_found = false;
    for (const auto& entry : std::filesystem::directory_iterator(temp_log_dir_)) {
        if (entry.path().filename().string().find("sgdlib") == 0) {
            log_found = true;

            // check contents
            std::ifstream log_file(entry.path());
            std::string content((std::istreambuf_iterator<char>(log_file)),
                                std::istreambuf_iterator<char>());
            EXPECT_TRUE(content.find("Logger initialized successfully") != std::string::npos);
            break;
        }
    }
    EXPECT_TRUE(log_found) << "No log file found in " << temp_log_dir_;
}

