#ifndef CORE_LOSS_LOG_LOSS_HPP_
#define CORE_LOSS_LOG_LOSS_HPP_

#include "base.hpp"

namespace sgdlib {
namespace detail {

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
    ~LogLoss() = default;

    FeatValType evaluate(const FeatValType& y_pred,
                         const LabelValType& y_true) const override {
        const FeatValType z = y_pred * static_cast<FeatValType>(y_true);
        if (z > 18.0) {
            return std::exp(-z);
        }
        if (z < -18.0) {
            return -z;
        }
        // numerically stable log(1 + exp(-z))
        return std::log1p(std::exp(-z));
    }

    FeatValType derivate(const FeatValType& y_pred,
                         const LabelValType& y_true) const override {
        const FeatValType z = y_pred * static_cast<FeatValType>(y_true);
        if (z > 18.0) {
            return std::exp(-z) * (-static_cast<FeatValType>(y_true));
        }
        if (z < -18.0) {
            return -static_cast<FeatValType>(y_true);
        }
        return -static_cast<FeatValType>(y_true) / (std::exp(z) + 1.0);
    }

    FeatValType evaluate_with_gradient(
        const std::vector<FeatValType>& X,
        const std::vector<LabelValType>& y,
        const std::vector<FeatValType>& w,
        std::vector<FeatValType>& grad) const override {

        const std::size_t num_samples = y.size();
        const std::size_t num_features = X.size() / num_samples;
        const FeatValType inv_num_samples = 1.0 / static_cast<FeatValType>(num_samples);
        FeatValType loss = 0.0;
        FeatValType dloss, y_hat;

#if defined(USE_OPENMP)
        // OpenMP reduction for loss and manual reduction for grad
        #pragma omp parallel reduction(+:loss) private(dloss, y_hat)
        {
            // define independant local_grad for each thread
            std::vector<FeatValType> local_grad(num_features, 0.0);

            #pragma omp for nowait
            for (std::size_t i = 0; i < num_samples; ++i) {
                y_hat = sgdlib::detail::vecdot<FeatValType>(
                    X.data() + (i * num_features),
                    X.data() + ((i + 1) * num_features),
                    w.data()
                );

                loss += evaluate(y_hat, y[i]);
                dloss = derivate(y_hat, y[i]);
                for (std::size_t j = 0; j < num_features; ++j) {
                    local_grad[j] += dloss * X[i * num_features + j];
                }
            }

            // reduction for grad and loss
            #pragma omp critical
            {
                for (std::size_t j = 0; j < num_features; ++j) {
                    grad[j] += local_grad[j];
                }
            }
        }
#else
        for (std::size_t i = 0; i < num_samples; ++i) {
            // compute W * X
            y_hat = sgdlib::detail::vecdot<FeatValType>(
                X.data() + (i * num_features),
                X.data() + ((i + 1) * num_features),
                w.data()
            );

            loss += evaluate(y_hat, y[i]);
            dloss = derivate(y_hat, y[i]);
            for (std::size_t j = 0; j < num_features; ++j) {
                grad[j] += dloss * X[i * num_features + j];
            }
        }
#endif
        loss *= inv_num_samples;
        sgdlib::detail::vecscale<FeatValType>(grad, inv_num_samples, grad);

        return loss;
    }
};

} // namespace detail
} // namespace sgdlib

#endif // CORE_LOSS_LOG_LOSS_HPP_
