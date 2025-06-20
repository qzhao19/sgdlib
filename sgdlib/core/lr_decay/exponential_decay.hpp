#ifndef CORE_LR_DECAY_EXPONENTIAL_DECAY_HPP_
#define CORE_LR_DECAY_EXPONENTIAL_DECAY_HPP_

#include "base.hpp"

namespace sgdlib {
namespace detail {

class Exponential final: public LRDecay{
public:
    Exponential(sgdlib::LRDecayParamType lr_decay_param): LRDecay(lr_decay_param) {};
    ~Exponential() = default;;

    /**
     * eta = eta0 * exp(-gamma * epoch)
    */
    sgdlib::ScalarType compute(std::size_t epoch) override {
        return lr_decay_param_["eta0"] * \
            std::exp((-lr_decay_param_["gamma"]) * static_cast<sgdlib::ScalarType>(epoch));
    }
};

} // namespace detail
} // namespace sgdlib

#endif // CORE_LR_DECAY_EXPONENTIAL_DECAY_HPP_
