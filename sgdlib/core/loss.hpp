#ifndef CORE_LOSS_HPP_
#define CORE_LOSS_HPP_

#include "loss/log_loss.hpp"
#include "loss/huber_loss.hpp"

// DEFINE_SHARED_REGISTRY(LossFunctionRegistry, sgdlib::detail::LossFunction, LossParamType);

namespace sgdlib {
namespace detail {

REGISTER_CLASS(LossFunctionRegistry, LogLoss, sgdlib::detail::LogLoss);
REGISTER_CLASS(LossFunctionRegistry, HuberLoss, sgdlib::detail::HuberLoss);

} // namespace detail
} // namespace sgdlib


#endif // CORE_LOSS_HPP_