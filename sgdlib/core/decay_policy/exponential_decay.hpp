#ifndef CORE_DECAY_POLICY_EXPONENTIAL_DECAY_HPP_
#define CORE_DECAY_POLICY_EXPONENTIAL_DECAY_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "base.hpp"

namespace sgdlib {
namespace internal {

class ExponentialDecay: public LRDecay{
public:
    ExponentialDecay(double eta0, 
                     double decay): LRDecay(eta0, decay) {};
    ~ExponentialDecay() {};

    /**
     * eta = eta0 * exp(-decay * epoch) 
    */
    double compute(std::size_t epoch) override {
        return this->eta0_ * std::exp((-this->decay_) * static_cast<double>(epoch));
    }
};

} // namespace internal
} // namespace sgdlib


#endif // CORE_DECAY_POLICY_EXPONENTIAL_DECAY_HPP_