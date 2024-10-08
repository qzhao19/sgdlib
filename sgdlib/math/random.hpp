#ifndef MATH_RANDOM_HPP_
#define MATH_RANDOM_HPP_

#include "common/prereqs.hpp"

namespace sgdlib {
namespace internal {

class RandomState final {
protected:
    // random eigine 
    std::mt19937  engine_;

public:
    // Create and initialize random number generator with the current system time
    RandomState(): engine_(static_cast<unsigned long>(time(nullptr))) {};

    /**
     * Create and initialize random number generator with a seed
     * @param seed
    */
    RandomState(unsigned long seed): engine_(seed) {};

    /**
     * Provide a double random number from a uniform distribution between [low, high).
     * @param low included lower bound for random number.
     * @param high excluded upper bound for random number.
     * @return random number.
    */
    double uniform_real(double low,
                        double high) {
        std::uniform_real_distribution<double> dist(low, high);
        return dist(engine_);
    };

    /**
     * Provide a long random number from a uniform distribution between [low, high).
     * @param low  included lower bound for random number.
     * @param high excluded upper bound for random number.
     * @return random number.
    */
    long uniform_int(long low,
                     long high) {
        
        std::uniform_int_distribution<long> dist(low, high - 1);
        return dist(engine_);          
    };

    /**
     * Shuffles the elements of a vector randomly.
     * 
     * @param x A reference to the vector of type `Type` to be shuffled.
    */
    template<typename Type>
    void shuffle(std::vector<Type>& x) {
        std::shuffle(std::begin(x), std::end(x), engine_);
    };

    /**
     * Randomly extract one element from x without repetition.
    */
    template<typename Type>
    Type sample(std::vector<Type>& x) {

        std::size_t size = x.size();
        std::uniform_int_distribution<long> dist(0, size - 1);

        // randomly generate an index of x
        std::size_t index = dist(engine_);
        
        // swap selected value with the last element of x
        // std::swap(x[index], x.back());
        // --size;

        return x[index];
    }

    /** 
     * Randomlt generate an index from the range [low, high).
    */
    std::size_t random_index(std::size_t low, 
                             std::size_t high) {
        std::uniform_int_distribution<std::size_t> dist(low, high - 1);
        return dist(engine_); 
    }

}; 

} // namespace internal
} // namespace sgdlib

#endif // COMMON_RANDOM_HPP_