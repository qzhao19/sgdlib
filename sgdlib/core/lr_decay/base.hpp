#ifndef CORE_LR_DECAY_BASE_HPP_
#define CORE_LR_DECAY_BASE_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "common/registry.hpp"

namespace sgdlib {

class LRDecay {
protected:
    double eta0_;
    double decay_;

public:
    LRDecay(double eta0, double decay): eta0_(eta0), decay_(decay) {};
    ~LRDecay() {};

    virtual double compute(std::size_t epoch) = 0;
};

// Create registries for base LR Decay function
DECLARE_REGISTRY(LRDecayRegistry, LRDecay, double, double);
DEFINE_REGISTRY(LRDecayRegistry, LRDecay, double, double);

} // namespace sgdlib

#endif // CORE_LR_DECAY_BASE_HPP_