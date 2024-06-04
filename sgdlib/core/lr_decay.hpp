#ifndef CORE_LR_DECAY_HPP_
#define CORE_LR_DECAY_HPP_

#include "lr_decay/exponential_decay.hpp"
#include "lr_decay/invscaling_decay.hpp"

namespace sgdlib {

// Create registries for log loss function
REGISTER_CLASS(LRDecayRegistry, Invscaling, Invscaling);
// Create registries for log loss function
REGISTER_CLASS(LRDecayRegistry, Exponential, Exponential);

}

#endif // CORE_LR_DECAY_HPP_