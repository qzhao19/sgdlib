#ifndef CORE_LR_DECAY_EXPONENTIAL_DECAY_HPP_
#define CORE_LR_DECAY_EXPONENTIAL_DECAY_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "base.hpp"

namespace sgdlib {

class Exponential final: public LRDecay{
public:
    Exponential(LRDecayParamType lr_decay_param): LRDecay(lr_decay_param) {};
    ~Exponential() {};

    /**
     * eta = eta0 * exp(-gamma * epoch) 
    */
    double compute(std::size_t epoch) override {
        return this->lr_decay_param_.at("eta0") * \
            std::exp((-this->lr_decay_param_.at("gamma")) * static_cast<double>(epoch));
    }
};

} // namespace sgdlib

#endif // CORE_LR_DECAY_EXPONENTIAL_DECAY_HPP_