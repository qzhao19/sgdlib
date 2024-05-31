#ifndef ALGORITHM_SGD_SGD_HPP_
#define ALGORITHM_SGD_SGD_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "algorithm/base.hpp"
#include "core/loss/base.hpp"
#include "core/decay_policy/base.hpp"
#include "core/decay_policy/invscaling_decay.hpp"
#include "core/decay_policy/exponential_decay.hpp"

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
        std::string penalty,
        std::string lr_policy,
        double alpha,
        double eta0,
        double tol,
        double decay,
        std::size_t max_iters, 
        std::size_t batch_size,
        std::size_t num_iters_no_change,
        std::size_t random_seed,
        bool fit_intercept = true,
        bool multi_class = false,
        bool shuffle = true, 
        bool verbose = true): BaseOptimizer(x0, 
            loss, penalty, 
            lr_policy, alpha, 
            eta0, tol, 
            decay,
            max_iters, 
            batch_size, 
            num_iters_no_change,
            random_seed,
            fit_intercept,
            multi_class,
            shuffle, 
            verbose) {};
    ~SGD () {};

    void optimize(const std::vector<FeatureType>& X, 
                  const std::vector<LabelType>& y) override {
        
        std::size_t num_samples = y.size();
        std::size_t num_features = X.size() / y.size();

        // std::size_t update_num_features;
        if (this->fit_intercept_) {
            this->num_features_ = num_features + 1;
        }
        else {
            this->num_features_ = num_features;
        }

        std::size_t step_per_iter = num_samples / this->batch_size_;
        std::size_t no_improvement_count = 0;

        bool is_converged = false;
        double best_loss = INF;

        // initialize a lookup table for training X, y
        std::vector<std::size_t> X_data_index(num_samples);
        std::iota(X_data_index.begin(), X_data_index.end(), 0);

        // initialize loss, gradient, x0 (weight)
        double loss;
        std::vector<FeatureType> grad(this->num_features_, 0.0);
        std::vector<FeatureType> x0(this->num_features_, 1.0);
        std::copy(this->x0_.begin(), this->x0_.end(), x0.begin());

        // initialize X_batch, y_batch and batch_data_index
        std::vector<FeatureType> X_batch(this->num_features_*this->batch_size_, 1.0);
        std::vector<FeatureType> y_batch(this->batch_size_, 0);
        std::vector<std::size_t> batch_data_index(this->batch_size_);
        
        // initialize loss function 
        std::unique_ptr<sgdlib::LossFunction> loss_fn = LossFunctionRegistry()->Create(loss, 0.0);

        // initialize learning rate scheduler
        if (lr_policy == "invscaling") {
            lr_decay = std::make_shared<sgdlib::internal::InvscalingDecay>(
                this->eta0_, this->decay_
            );
        }
        else if (lr_policy == "exponential") {
            lr_decay = std::make_shared<sgdlib::internal::ExponentialDecay>(
                this->eta0_, this->decay_
            );
        }

        // 
        for (std::size_t iter = 0; iter < this->max_iters_; iter++) {
            // enable to shuffle mask of data for on batch
            if (this->shuffle_) {
                this->random_state_.shuffle<FeatureType>(X_data_index);
            }

            // apply lr decay policy to compute eta
            double eta = lr_decay->compute(iter);
            for (std::size_t i = 0; i < step_per_iter; ++i) {
                // copy batch data indices to batch_data_index
                std::copy(&X_data_index[i] * this->batch_size_, 
                          (&X_data_index[i] + 1) * this->batch_size_, 
                          batch_data_index);
                
                // copy X_batch and y_batch data
                for (std::size_t j = 0; j < this->batch_size_; ++j) {
                    std::copy(&X[batch_data_index[j] * num_features], 
                              &X[(batch_data_index[j] + 1) * num_features], 
                              X_batch.begin() + (j * this->num_features_));
                    y_batch[j] = y[batch_data_index[j]];
                };

                loss = loss_fn->evaluate(X_batch, y_batch, x0);

                loss_fn->gradient(X_batch, y_batch, x0, grad);

                // gradient clipping
                sgdlib::internal::clip<FeatureType>(grad, MIN_DLOSS, MAX_DLOSS);

                // update x0
                

            }


        }
    }

};

} // namespace sgdlib

#endif // ALGORITHM_SGD_SGD_HPP_