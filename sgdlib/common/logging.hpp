#ifndef COMMON_LOGGING_HPP_
#define COMMON_LOGGING_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {
namespace detail {

/**
 * @brief Prints runtime information at specified intervals.
 *
 * This function prints a message composed of the provided arguments every 'print_interval'
 * number of calls. It uses a static counter to keep track of the number of calls.
 *
 * @tparam Args Variadic template parameter for the types of arguments to be printed.
 * @param print_interval The interval at which to print the information. The function
 *                       will print every 'print_interval' calls.
 * @param args Variable number of arguments to be included in the printed message.
 *             These can be of any type that can be converted to a string.
 *
 * @note This function does not return a value. It prints to the standard output.
 */
template<typename... Args>
void print_runtime_info(int print_interval, Args... args) {
    static int counter = 0;  // records the number of calls
    counter++;  //increments counter

    // prints message if counter is a multiple of print_interval
    if (counter % print_interval == 0) {
        std::ostringstream oss;
        (oss << ... << args);  // concatenates all the arguments
        std::cout << "INFO: " << oss.str() << std::endl;
    }
}

/**
 * @brief Throws a custom exception with a formatted error message.
 *
 * This function constructs an error message by concatenating all provided arguments
 * and throws an exception of type 'ExceptionType' with the resulting message.
 *
 * @tparam ExceptionType The type of exception to be thrown. Must be a class derived from std::exception.
 * @tparam Args Variadic template parameter for the types of arguments.
 * @param args Variable number of arguments to be included in the error message.
 *             These can be of any type that can be inserted into an output stream.
 *
 * @throws ExceptionType Always throws this exception with the formatted error message.
 */
template <typename ExceptionType, typename... Args>
void throw_error_msg(Args... args) {
    std::ostringstream err_msg;
    err_msg << "ERROR: ";
    (err_msg << ... << args) << std::endl;
    throw ExceptionType(err_msg.str());
}


class Logger {
private:
    inline static std::atomic<bool> initialized_{false};
    inline static std::once_flag init_flag_;
    inline static std::string log_name_;

    //
    static void initialize_impl(const std::string& log_name = "sgdlib",
                                const std::filesystem::path& log_dir = "./logs",
                                bool log_to_stderr = false) {
        try {
            // check and create the log dir
            if (!log_dir.empty()) {
                std::error_code ecode;
                bool dir_exist = std::filesystem::create_directories(log_dir, ecode);
                if (ecode || !std::filesystem::exists(log_dir)) {
                    std::cerr << "Failed to create log directory: " << ecode.message() << "\n";
                    throw std::runtime_error("Log directory creation failed");
                }

                if (!std::filesystem::is_directory(log_dir)) {
                    throw std::runtime_error("Log path is not a directory");
                }
            }
            // setup glog params
            FLAGS_log_dir = log_dir.string();
            FLAGS_logtostderr = log_to_stderr;
            FLAGS_minloglevel = 0;

            // init glog
            google::InitGoogleLogging(log_name.c_str());
            google::InstallFailureSignalHandler();

            // update
            log_name_ = log_name;
            initialized_.store(true, std::memory_order_release);

            // write starting info
            LOG(INFO) << "Logger initialized successfully. Log directory: " << log_dir;
        } catch (const std::exception& error) {
            std::cerr << "Logger initialization failed: " << error.what() << std::endl;
            initialized_.store(false, std::memory_order_release);
            throw;
        }
    }

    struct CleanupGuard {
        ~CleanupGuard() {
            if (Logger::initialized_.load(std::memory_order_acquire)) {
                google::ShutdownGoogleLogging();
            }
        }
    };
    inline static CleanupGuard cleanup_;

public:
    // delete copy and operator const
    Logger() = delete;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    static bool is_initialized() noexcept {
        return initialized_.load(std::memory_order_acquire);
    }

    static void initialize(const std::string& log_name = "sgdlib",
                           const std::string& log_dir = "./logs",
                           bool log_to_stderr = false) {
        std::call_once(init_flag_, initialize_impl, log_name, log_dir, log_to_stderr);
    }

    // keep following implementation:
    // init_flag_ = std::once_flag() is illegal
    // static void reset() {
    //     if (is_initialized()) {
    //         google::ShutdownGoogleLogging();

    //     }
    //     initialized_.store(false, std::memory_order_release);
    //     init_flag_ = std::once_flag();
    // }
};

} // namespace detail
} // namespace sgdlib

#define PRINT_RUNTIME_INFO(print_interval, ...) \
    sgdlib::detail::print_runtime_info(print_interval, __VA_ARGS__)

#define THROW_RUNTIME_ERROR(...) sgdlib::detail::throw_error_msg<std::runtime_error>(__VA_ARGS__)

#define THROW_INVALID_ERROR(...) sgdlib::detail::throw_error_msg<std::invalid_argument>(__VA_ARGS__)

#define THROW_LOGIC_ERROR(...) sgdlib::detail::throw_error_msg<std::logic_error>(__VA_ARGS__)

#define THROW_OUT_RANGE_ERROR(...) sgdlib::detail::throw_error_msg<std::out_of_range>(__VA_ARGS__)

#define SGDLIB_LOG(severity) \
    (sgdlib::detail::Logger::is_initialized() ? \
        LOG(severity) : \
        (sgdlib::detail::Logger::initialize(), LOG(severity)))


#endif //COMMON_LOGGING_HPP_
