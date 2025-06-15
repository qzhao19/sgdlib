#ifndef CORE_LOSS_LOG_LOSS_HPP_
#define CORE_LOSS_LOG_LOSS_HPP_

#include "base.hpp"

namespace sgdlib {
namespace detail {

/**
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
    LogLoss(sgdlib::LossParamType loss_param): LossFunction(loss_param) {};
    ~LogLoss() = default;

    sgdlib::FeatureType evaluate(const sgdlib::FeatureType &y_pred,
                                 const sgdlib::LabelType &y_true) const override {
        const sgdlib::FeatureType z = y_pred * static_cast<sgdlib::FeatureType>(y_true);
        if (z > 18.0) {
            return std::exp(-z);
        }
        if (z < -18.0) {
            return -z;
        }
        // numerically stable log(1 + exp(-z))
        return std::log1p(std::exp(-z));
    }

    sgdlib::FeatureType derivate(const sgdlib::FeatureType &y_pred,
                                 const sgdlib::LabelType &y_true) const override {
        const sgdlib::FeatureType y_true_float = static_cast<sgdlib::FeatureType>(y_true);
        const sgdlib::FeatureType z = y_pred * y_true_float ;
        if (z > 18.0) {
            return std::exp(-z) * (-y_true_float);
        }
        if (z < -18.0) {
            return -y_true_float;
        }
        return -y_true_float / (std::exp(z) + 1.0);
    }

    sgdlib::FeatureType evaluate_with_gradient(const sgdlib::ArrayDatasetType &dataset,
                                               const std::vector<sgdlib::FeatureType> &w,
                                               std::vector<sgdlib::FeatureType> &grad) const override {
        // get num samples and features
        const std::size_t num_samples = dataset.nrows();
        const std::size_t num_features = dataset.ncols();
        const sgdlib::FeatureType inv_num_samples = 1.0 / static_cast<sgdlib::FeatureType>(num_samples);
        sgdlib::FeatureType loss = 0.0;
        sgdlib::FeatureType dloss, y_hat;
        this->dloss_history_.resize(num_samples);

#if defined(USE_OPENMP)
        int num_threads = 1;
        // get_num_threads could be risk
        #pragma omp parallel
        {
            #pragma omp single
            num_threads = omp_get_max_threads();
        }

        // define independant local_grad for each thread
        std::vector<std::vector<sgdlib::FeatureType>> local_grad(
            num_threads, std::vector<sgdlib::FeatureType>(num_features, 0.0)
        );

        // OpenMP reduction for loss and manual reduction for grad
        #pragma omp parallel reduction(+:loss) private(dloss, y_hat)
        {
            int thread_id = omp_get_thread_num();
            std::vector<sgdlib::FeatureType>& thread_grad = local_grad[thread_id];

            sgdlib::LabelType y;
            std::vector<sgdlib::FeatureType> x(num_features);
            #pragma omp for nowait
            for (std::size_t i = 0; i < num_samples; ++i) {
                // get x_i, y_i
                dataset.X_row_data(i, x);
                dataset.y_row_data(i, y);
                y_hat = sgdlib::detail::vecdot<sgdlib::FeatureType>(x, w);
                loss += evaluate(y_hat, y);
                dloss = derivate(y_hat, y);
                this->dloss_history_[i] = dloss;
                sgdlib::detail::vecadd<sgdlib::FeatureType>(x, dloss, thread_grad);
            }
        }
        // main thread reduction for grad
        for (std::size_t t = 0; t < num_threads; ++t) {
            sgdlib::detail::vecadd<sgdlib::FeatureType>(grad, local_grad[t], grad);
        }
#else
        sgdlib::LabelType y;
        std::vector<sgdlib::FeatureType> x(num_features);
        for (std::size_t i = 0; i < num_samples; ++i) {
            // get x_i, y_i
            dataset.X_row_data(i, x);
            dataset.y_row_data(i, y);
            // compute W * X
            y_hat = sgdlib::detail::vecdot<sgdlib::FeatureType>(x, w);
            loss += evaluate(y_hat, y);
            dloss = derivate(y_hat, y);
            this->dloss_history_[i] = dloss;
            sgdlib::detail::vecadd<sgdlib::FeatureType>(x, dloss, grad);
        }
#endif
        loss *= inv_num_samples;
        sgdlib::detail::vecscale<sgdlib::FeatureType>(grad, inv_num_samples, grad);

        if (callback_) {
            callback_(this->dloss_history_);
        }
        return loss;
    }
};

} // namespace detail
} // namespace sgdlib

#endif // CORE_LOSS_LOG_LOSS_HPP_
