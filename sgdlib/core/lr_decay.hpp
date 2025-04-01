#ifndef CORE_LR_DECAY_HPP_
#define CORE_LR_DECAY_HPP_

#include "lr_decay/exponential_decay.hpp"
#include "lr_decay/invscaling_decay.hpp"

namespace sgdlib {
namespace detail {

REGISTER_CLASS(LRDecayRegistry, Invscaling, Invscaling);
REGISTER_CLASS(LRDecayRegistry, Exponential, Exponential);

} // namespace detail
} // namespace sgdlib

#endif // CORE_LR_DECAY_HPP_