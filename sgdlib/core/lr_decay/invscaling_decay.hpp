#ifndef CORE_LR_DECAY_INVSCALING_DECAY_HPP_
#define CORE_LR_DECAY_INVSCALING_DECAY_HPP_

#include "base.hpp"

namespace sgdlib {

class Invscaling final: public LRDecay{
public:
    Invscaling(LRDecayParamType lr_decay_param): LRDecay(lr_decay_param) {};
    ~Invscaling() {};

    /**
     * eta = eta0 / pow(epoch, gamma) 
    */
    double compute(std::size_t epoch) override {
        return lr_decay_param_.at("eta0") / std::pow(epoch + 1, lr_decay_param_.at("gamma"));
    }
};

} // namespace sgdlib

#endif // CORE_LR_DECAY_INVSCALING_DECAY_HPP_