#ifndef COMMON_LOGGING_HPP_
#define COMMON_LOGGING_HPP_

#include "common/prereqs.hpp"

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
 * @brief Throws a runtime error with a formatted error message.
 *
 * This function constructs an error message by concatenating all provided arguments
 * and throws a std::runtime_error with the resulting message.
 *
 * @tparam Args Variadic template parameter for the types of arguments.
 * @param args Variable number of arguments to be included in the error message.
 *             These can be of any type that can be inserted into an output stream.
 *
 * @throws std::runtime_error Always throws this exception with the formatted error message.
 */
template <typename... Args>
void throw_runtime_error(Args... args) {
    std::ostringstream err_msg;
    err_msg << "ERROR: ";
    (err_msg << ... << args) << std::endl;
    throw std::runtime_error(err_msg.str());
}

/**
 * @brief Throws an invalid argument error with a formatted error message.
 *
 * This function constructs an error message by concatenating all provided arguments
 * and throws a std::invalid_argument exception with the resulting message.
 *
 * @tparam Args Variadic template parameter for the types of arguments.
 * @param args Variable number of arguments to be included in the error message.
 *             These can be of any type that can be inserted into an output stream.
 *
 * @throws std::invalid_argument Always throws this exception with the formatted error message.
 */
template <typename... Args>
void throw_invalid_arg_error(Args... args) {
    std::ostringstream err_msg;
    err_msg << "ERROR: ";
    (err_msg << ... << args) << std::endl;
    throw std::invalid_argument(err_msg.str());
}

#define PRINT_RUNTIME_INFO(print_interval, ...) \
    print_runtime_info(print_interval, __VA_ARGS__)

#define THROW_RUNTIME_ERROR(...) throw_runtime_error(__VA_ARGS__)

#define THROW_INVALID_ARG_ERROR(...) throw_invalid_arg_error(__VA_ARGS__)


#endif //COMMON_LOGGING_HPP_