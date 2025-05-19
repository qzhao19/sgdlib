#ifndef CORE_LR_DECAY_BASE_HPP_
#define CORE_LR_DECAY_BASE_HPP_

#include "common/constants.hpp"
#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "common/registry.hpp"

namespace sgdlib {
namespace detail {

/**
 * @file base.hpp
 *
 * @class LRDecay
 *
 * @brief Abstract base class representing learning rate decay
 *
*/
class LRDecay {
protected:
    LRDecayParamType lr_decay_param_;

public:
    LRDecay(LRDecayParamType lr_decay_param): lr_decay_param_(lr_decay_param) {};
    virtual ~LRDecay() = default;
    virtual FloatType compute(std::size_t epoch) = 0;
};

// Create registries for base LR Decay function
DECLARE_UNIQUE_REGISTRY(LRDecayRegistry, LRDecay, LRDecayParamType);
DEFINE_UNIQUE_REGISTRY(LRDecayRegistry, LRDecay, LRDecayParamType);

} // namespace detail
} // namespace sgdlib

#endif // CORE_LR_DECAY_BASE_HPP_
