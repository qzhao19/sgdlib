#ifndef CORE_LOSS_LOG_LOSS_HPP_
#define CORE_LOSS_LOG_LOSS_HPP_

#include "base.hpp"

namespace sgdlib {

/**
 * @file log_loss.hpp
 * 
 * @class LogLoss
 * 
 * @brief logistic regression loss function for binary classification 
 * with y in {-1, 1}. An approximation is used to simplify calculations, 
 * specifically avoiding the computation of a logarithm, it can help 
 * maintain numerical stability by avoiding extreme values.
 * 
 * Here, we use another way to express loss function with the categories 
 * as {-1, 1}, let x_i be the i-th feature vector, w be the parameter 
 * vector for the logistic regression, N be the sample size, and p(y_i) 
 * be the predicted probability of membership to category 1
 * so p(y_i) = p_i = w * x_i
 * 
 *  L = (1/N) * sum(log(1.0 + exp(-y_i * p_i)))
 * 
 *  dL/dp = (1/N) * sum(-y / (1 + exp(y_i * p_i)))
 * 
*/
class LogLoss final: public LossFunction {
public:
    LogLoss(LossParamType loss_param): LossFunction(loss_param) {};
    ~LogLoss() {};

    virtual FeatValType evaluate(const FeatValType& y_pred, 
                                 const LabelValType& y_true) const override {
        
        FeatValType z = y_pred * static_cast<FeatValType>(y_true);
        if (z > 18.0) {
            return std::exp(-z);
        }
        if (z < -18.0) {
            return -z;
        }
        return std::log(1.0 + std::exp(-z));
    }

    virtual FeatValType derivate(const FeatValType& y_pred, 
                                 const LabelValType& y_true) const override {

        FeatValType z = y_pred * static_cast<FeatValType>(y_true);
        if (z > 18.0) {
            return std::exp(-z) * (-static_cast<FeatValType>(y_true));
        }
        if (z < -18.0) {
            return -static_cast<FeatValType>(y_true);
        }
        return -static_cast<FeatValType>(y_true) / (std::exp(z) + 1.0);
    }

};

} // namespace sgdlib

#endif // CORE_LOSS_LOG_LOSS_HPP_