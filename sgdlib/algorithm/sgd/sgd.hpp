#ifndef ALGORITHM_SGD_SGD_HPP_
#define ALGORITHM_SGD_SGD_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "algorithm/base.hpp"
#include "core/loss/base.hpp"
#include "core/loss/log_loss.hpp"

namespace sgdlib {

/**
 * @file sgd.hpp
 * 
 * @brief A stochastic gradient descent classifier.
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
        double power_t,
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
            power_t,
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

        std::size_t new_num_features;
        if (this->fit_intercept_) {
            new_num_features = num_features + 1;
        }
        else {
            new_num_features = num_features;
        }

        std::size_t step_per_iter = num_samples / this->batch_size_;
        std::size_t no_improvement_count = 0;

        bool is_converged = false;
        double best_loss = INF;

        // initialize a lookup table for training X, y
        std::vector<std::size_t> sample_indices(this->batch_size_);
        std::iota(sample_indices.begin(), sample_indices.end(), 0);

        // initialize weight, X_batch, y_batch
        std::vector<FeatureType> w(new_num_features, 1.0);
        std::vector<FeatureType> X_batch(new_num_features*this->batch_size_, 1.0), y_batch(this->batch_size_, 0);
        
        // initialize loss function 
        std::unique_ptr<sgdlib::LossFunction> loss_fn = LossFunctionRegistry()->Create("LogLoss", 0.0);

        for (std::size_t iter = 0; iter < this->max_iters_; iter++) {
            // enable to shuffle mask of data for on batch
            if (this->shuffle_) {
                this->random_state_.shuffle<FeatureType>(sample_indices);
            }

            // copy X, y to X_batch and y_batch
            for (std::size_t i = 0; i < step_per_iter; ++i) {
                std::copy(&X[sample_indices[i] * num_features], 
                          &X[(sample_indices[i] + 1) * num_features], 
                          X_batch.begin() + (i * new_num_features));
                y_batch[i] = y[sample_indices[i]];
            }

            loss_fn->gradient(X_batch, y_batch, w, grad);



        }
    }

};

} // namespace sgdlib

#endif // ALGORITHM_SGD_SGD_HPP_