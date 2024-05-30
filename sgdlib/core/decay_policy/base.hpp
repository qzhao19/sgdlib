#ifndef CORE_DECAY_POLICY_BASE_HPP_
#define CORE_DECAY_POLICY_BASE_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"

namespace sgdlib {
namespace internal {

class LRDecay {
protected:
    double eta0_;
    double decay_;

public:
    LRDecay(double eta0, double decay): eta0_(eta0), decay_(decay) {};
    ~LRDecay() {};

    virtual double compute(std::size_t epoch) = 0;
};

} // namespace internal
} // namespace sgdlib

#endif // CORE_DECAY_POLICY_BASE_HPP_