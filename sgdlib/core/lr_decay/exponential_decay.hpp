#ifndef CORE_LR_DECAY_EXPONENTIAL_DECAY_HPP_
#define CORE_LR_DECAY_EXPONENTIAL_DECAY_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "base.hpp"

namespace sgdlib {

class Exponential: public LRDecay{
public:
    Exponential(double eta0, 
                     double decay): LRDecay(eta0, decay) {};
    ~Exponential() {};

    /**
     * eta = eta0 * exp(-decay * epoch) 
    */
    double compute(std::size_t epoch) override {
        return this->eta0_ * std::exp((-this->decay_) * static_cast<double>(epoch));
    }
};

} // namespace sgdlib

#endif // CORE_LR_DECAY_EXPONENTIAL_DECAY_HPP_