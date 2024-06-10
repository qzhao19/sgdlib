#ifndef ALGORITHM_SGD_SGD_HPP_
#define ALGORITHM_SGD_SGD_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

/**
 * @file sgd.hpp
 * 
 * @brief Stochastic Gradient Descent (SGD) optimizer.
 * 
 * @param x0 FeatureType. Initial parameter vector.
 *      This vector represents the starting point of the optimization process.
 * @param loss String. The type of loss function to be used for the optimization.
 *      Common choices include "log_loss" for classification tasks.
 * @param lr_policy String
 *      The learning rate schedule policy, which can be "exponential" or "invscaling".
 * @param alpha 
 *      The regularization strength, which penalizes large parameter values to prevent overfitting.
 *      The higher the alpha, the stronger the regularization.
 * @param eta0 Double. The initial value of the learning rate.
 * @param tol Double The convergence tolerance, which is a threshold below which the change in the loss function
 *      indicates that the optimizer has likely converged.
 * @param gamma Double A hyperparameter that controls the learning rate policy.
 * @param max_iters Integer. The maximum number of iterations (epochs) that the optimizer will run before stopping.
 * @param batch_size Integer. The number of samples to be processed in each iteration of the optimization loop.
 *      Smaller batch sizes can provide a regularizing effect, while larger batch sizes may 
 *      offer computational efficiency.
 * @param num_iters_no_change Integer. The number of iterations after which the optimizer will stop if there is no
 *      improvement in the loss function.
 * @param random_seed Integer The seed for the random number generator, which ensures reproducibility of the results
 *      when shuffling the data or selecting batches.
 * @param shuffle Boolean. Whether to randomly shuffle the data before starting each iteration. This can help prevent
 *      the optimizer from converging to a local minimum and is generally recommended.
 * @param verbose Boolean Whether to output detailed information during the optimization process, such as the progress,
 *      loss values, and parameter updates.
 *
*/
class SGD: public BaseOptimizer {
public:
    SGD(const std::vector<FeatureType>& x0, 
        std::string loss, 
        std::string lr_policy,
        double alpha,
        double eta0,
        double tol,
        double gamma,
        std::size_t max_iters, 
        std::size_t batch_size,
        std::size_t num_iters_no_change,
        std::size_t random_seed,
        bool shuffle = true, 
        bool verbose = true): BaseOptimizer(x0, 
            loss, lr_policy, 
            alpha, eta0, 
            tol, gamma,
            max_iters, 
            batch_size, 
            num_iters_no_change,
            random_seed,
            shuffle, 
            verbose) {};
    ~SGD() {};

    void optimize(const std::vector<FeatureType>& X, 
                  const std::vector<LabelType>& y) override {
        
        std::size_t num_samples = y.size();
        std::size_t num_features = x0_.size();

        std::size_t step_per_iter = num_samples / batch_size_;
        std::size_t no_improvement_count = 0;
        std::size_t iter = 0;

        bool is_converged = false;
        bool is_infinity = false;
        double best_loss = INF;

        // initialize a lookup table for training X, y
        std::vector<std::size_t> X_data_index(num_samples);
        std::iota(X_data_index.begin(), X_data_index.end(), 0);

        // initialize loss, loss_history, gradient, x0 (weight)
        double loss;
        std::vector<double> loss_history(step_per_iter, 0);
        std::vector<FeatureType> grad(num_features, 0.0);
        std::vector<FeatureType> x0 = x0_;

        // initialize X_batch, y_batch and batch_data_index
        std::vector<FeatureType> X_batch(num_features*batch_size_);
        std::vector<LabelType> y_batch(batch_size_);
        std::vector<std::size_t> batch_data_index(batch_size_);
        
        // 
        for (iter = 0; iter < max_iters_; ++iter) {
            // enable to shuffle mask of data for on batch
            if (shuffle_) {
                random_state_.shuffle<std::size_t>(X_data_index);
            }

            // apply lr decay policy to compute eta
            double eta = lr_decay_->compute(iter);
            for (std::size_t i = 0; i < step_per_iter; ++i) {
                // copy batch data indices to batch_data_index
                std::copy_n(X_data_index.begin() + (i * batch_size_), 
                            batch_size_, 
                            batch_data_index.begin());

                // copy X_batch and y_batch data
                for (std::size_t j = 0; j < batch_size_; ++j) {
                    std::copy_n(&X[batch_data_index[j] * num_features], 
                                num_features, 
                                X_batch.begin() + (j * num_features));
                    y_batch[j] = y[batch_data_index[j]];
                };

                // evaluate the loss on X_batch
                loss = loss_fn_->evaluate(X_batch, y_batch, x0);

                // compute gradient on X_batch
                loss_fn_->gradient(X_batch, y_batch, x0, grad);

                // gradient clipping
                sgdlib::internal::clip<FeatureType>(grad, MIN_DLOSS, MAX_DLOSS);

                // update x0: w = w - lr * grad
                for (std::size_t k = 0; k < num_features; ++k) {
                    x0[k] -= eta * grad[k]; 
                }

                // store loss value into loss_history
                loss_history[i] = loss;
            }

            // ---Convergence test---
            // check under/overflow
            if (sgdlib::internal::isinf(x0)) {
                is_infinity = true;
                break;
            }

            // compute sum of the loss value for one full batch
            double sum_loss = std::accumulate(loss_history.begin(), 
                                              loss_history.end(), 
                                              decltype(loss_history)::value_type(0));
            if ((tol_ > -INF) && (sum_loss > best_loss - tol_ * step_per_iter)) {
                no_improvement_count +=1;
            }
            else {
                no_improvement_count = 0;
            }
            if (sum_loss < best_loss) {
                best_loss = sum_loss;
            }

            // if there is no improvement is bigger than the threshold
            if (no_improvement_count >= num_iters_no_change_) {
                if (verbose_) {
                    std::cout << "Convergence after " << (iter + 1) << " epochs." << std::endl;
                }
                is_converged = true;
                break;
            }

            if (verbose_) {
                if ((iter % 1) == 0) {
                    std::cout << "Epoch = " << (iter + 1) << ", xnorm2 = " 
                              << sgdlib::internal::sqnorm2<FeatureType>(x0) << ", avg loss = " 
                              << sum_loss / static_cast<double>(step_per_iter) << std::endl;
                }
            }
        }

        if (is_infinity) {
            std::ostringstream err_msg;
            err_msg << "Floating-point under-/overflow occurred at epoch " << (iter + 1)
                    << ", try to scale input data with standard or minmax." << std::endl;
            throw std::runtime_error(err_msg.str());
        }

        if (!is_converged) {
            std::ostringstream err_msg;
            err_msg << "Not converge, current number of epoch = " << (iter + 1)
                    << ", the batch size = " << batch_size_ 
                    << ", try apply different parameters." << std::endl;
            throw std::runtime_error(err_msg.str());
        }

        x_opt_ = x0;
    }

};

} // namespace sgdlib

#endif // ALGORITHM_SGD_SGD_HPP_