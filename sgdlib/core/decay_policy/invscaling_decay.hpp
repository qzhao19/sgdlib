#ifndef CORE_DECAY_POLICY_INVSCALING_DECAY_HPP_
#define CORE_DECAY_POLICY_INVSCALING_DECAY_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "base.hpp"

namespace sgdlib {
namespace internal {

class InvscalingDecay: public LRDecay{
public:
    InvscalingDecay(double eta0, 
                    double decay): LRDecay(eta0, decay) {};
    ~InvscalingDecay() {};

    /**
     * eta = eta0 / pow(epoch, power_t) 
    */
    double compute(std::size_t epoch) override {
        return this->eta0_ / std::pow(epoch, this->decay_);
    }
};

} // namespace internal
} // namespace sgdlib

#endif // CORE_DECAY_POLICY_INVSCALING_DECAY_HPP_