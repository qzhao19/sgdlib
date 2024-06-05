#ifndef ALGORITHM_SGD_SGD_HPP_
#define ALGORITHM_SGD_SGD_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "algorithm/base.hpp"
#include "core/loss.hpp"
#include "core/lr_decay.hpp"

namespace sgdlib {

/**
 * @file sgd.hpp
 * 
 * @brief A stochastic gradient descent optimizer.
 * 
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
        double decay,
        std::size_t max_iters, 
        std::size_t batch_size,
        std::size_t num_iters_no_change,
        std::size_t random_seed,
        bool shuffle = true, 
        bool verbose = true): BaseOptimizer(x0, 
            loss, lr_policy, 
            alpha, eta0, 
            tol, decay,
            max_iters, 
            batch_size, 
            num_iters_no_change,
            random_seed,
            shuffle, 
            verbose) {};
    ~SGD () {};

    void optimize(const std::vector<FeatureType>& X, 
                  const std::vector<LabelType>& y) override {
        
        std::size_t num_samples = y.size();
        std::size_t num_features = this->x0_.size();

        std::size_t step_per_iter = num_samples / this->batch_size_;
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
        std::vector<FeatureType> x0 = this->x0_;

        // initialize X_batch, y_batch and batch_data_index
        std::vector<FeatureType> X_batch(num_features*this->batch_size_);
        std::vector<LabelType> y_batch(this->batch_size_);
        std::vector<std::size_t> batch_data_index(this->batch_size_);
        
        // initialize loss function 
        std::unique_ptr<sgdlib::LossFunction> loss_fn = LossFunctionRegistry()->Create(
            this->loss_, this->loss_params_
        );

        // initialize learning rate scheduler
        std::unique_ptr<sgdlib::LRDecay> lr_decay = LRDecayRegistry()->Create(
            this->lr_policy_, this->eta0_, this->decay_
        );

        // 
        for (iter = 0; iter < this->max_iters_; ++iter) {
            // enable to shuffle mask of data for on batch
            if (this->shuffle_) {
                this->random_state_.shuffle<std::size_t>(X_data_index);
            }

            // apply lr decay policy to compute eta
            double eta = lr_decay->compute(iter);
            for (std::size_t i = 0; i < step_per_iter; ++i) {
                // copy batch data indices to batch_data_index
                std::copy_n(X_data_index.begin() + (i * this->batch_size_), 
                            this->batch_size_, 
                            std::back_inserter(batch_data_index));
                
                // copy X_batch and y_batch data
                for (std::size_t j = 0; j < this->batch_size_; ++j) {
                    std::copy(&X[batch_data_index[j] * num_features], 
                              &X[(batch_data_index[j] + 1) * num_features], 
                              X_batch.begin() + (j * num_features));
                    y_batch[j] = y[batch_data_index[j]];
                };

                // evaluate the loss on X_batch
                loss = loss_fn->evaluate(X_batch, y_batch, x0);

                loss_fn->gradient(X_batch, y_batch, x0, grad);

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

            // evaluate the loss on the training dataset
            double sum_loss = std::accumulate(loss_history.begin(), loss_history.end(), 0);
            if ((this->tol_ > -INF) && (sum_loss > best_loss - this->tol_ * this->batch_size_)) {
                no_improvement_count +=1;
            }
            else {
                no_improvement_count = 0;
            }
            if (sum_loss < best_loss) {
                best_loss = sum_loss;
            }

            // 
            if (no_improvement_count >= this->num_iters_no_change_) {
                if (this->verbose_) {
                    std::cout << "Convergence after " << (iter + 1) << " epochs." << std::endl;
                }
                is_converged = true;
                break;
            }

            if (this->verbose_) {
                if ((iter % 1) == 0) {
                    std::cout << "-- Epoch = " << iter << ", average loss value = " 
                              << sum_loss / static_cast<double>(this->batch_size_) << std::endl;
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
                    << ", the batch size = " << this->batch_size_ 
                    << ", try apply different parameters." << std::endl;
            throw std::runtime_error(err_msg.str());
        }

        this->x_opt_ = x0;
    }

};

} // namespace sgdlib

#endif // ALGORITHM_SGD_SGD_HPP_