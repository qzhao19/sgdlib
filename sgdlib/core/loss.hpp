#ifndef CORE_LOSS_HPP_
#define CORE_LOSS_HPP_

#include "loss/log_loss.hpp"
#include "loss/huber_loss.hpp"

namespace sgdlib {

// Create registries for log loss function
REGISTER_CLASS(LossFunctionRegistry, LogLoss, LogLoss);
// Create registries for log loss function
REGISTER_CLASS(LossFunctionRegistry, HuberLoss, HuberLoss);

}

#endif // CORE_LOSS_HPP_