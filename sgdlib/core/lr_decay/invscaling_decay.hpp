#ifndef CORE_LR_DECAY_INVSCALING_DECAY_HPP_
#define CORE_LR_DECAY_INVSCALING_DECAY_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "base.hpp"

namespace sgdlib {

class Invscaling: public LRDecay{
public:
    Invscaling(double eta0, 
                    double decay): LRDecay(eta0, decay) {};
    ~Invscaling() {};

    /**
     * eta = eta0 / pow(epoch, power_t) 
    */
    double compute(std::size_t epoch) override {
        return this->eta0_ / std::pow(epoch, this->decay_);
    }
};

} // namespace sgdlib

#endif // CORE_LR_DECAY_INVSCALING_DECAY_HPP_