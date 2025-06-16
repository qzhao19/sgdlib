#ifndef CORE_LR_DECAY_INVSCALING_DECAY_HPP_
#define CORE_LR_DECAY_INVSCALING_DECAY_HPP_

#include "base.hpp"

namespace sgdlib {
namespace detail {

class Invscaling final: public LRDecay{
public:
    Invscaling(sgdlib::LRDecayParamType lr_decay_param): LRDecay(lr_decay_param) {};
    ~Invscaling() = default;

    /**
     * eta = eta0 / pow(epoch, gamma)
    */
    sgdlib::ScalarType compute(std::size_t epoch) override {
        return lr_decay_param_["eta0"] / std::pow(epoch + 1, lr_decay_param_["gamma"]);
    }
};

} // namespace detail
} // namespace sgdlib

#endif // CORE_LR_DECAY_INVSCALING_DECAY_HPP_
