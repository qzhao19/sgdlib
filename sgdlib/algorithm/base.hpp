#ifndef ALGORITHM_BASE_HPP_
#define ALGORITHM_BASE_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "math/random.hpp"
#include "core/loss.hpp"
#include "core/lr_decay.hpp"
#include "core/stepsize_search.hpp"

namespace sgdlib {

class BaseOptimizer {
protected:
    std::vector<FeatureType> w0_;
    FeatureType b0_;
    
    std::string loss_;
    std::string penalty_;
    std::string lr_policy_;

    double alpha_;
    double l1_ratio_;
    double eta0_;
    double tol_;
    double gamma_;

    std::size_t max_iters_;
    std::size_t batch_size_;
    std::size_t num_iters_no_change_;
    std::size_t random_seed_;

    bool is_saga_;
    bool shuffle_;
    bool verbose_;

    LossParamType loss_params_;
    LRDecayParamType lr_decay_params_;
    StepSizeSearchParamType stepsize_search_param_;

    std::vector<FeatureType> w_opt_;
    FeatureType b_opt_;
    sgdlib::internal::RandomState random_state_;

    std::unique_ptr<sgdlib::LossFunction> loss_fn_;
    std::unique_ptr<sgdlib::LRDecay> lr_decay_;

    void init_loss_params() {
        // initialize loss function 
        // margin threshold for hinge loss
        loss_params_["threshold"] = 1.0;
        loss_fn_ = LossFunctionRegistry()->Create(loss_, loss_params_);
    }

    void init_lr_params() {
        // initialize learning rate scheduler
        lr_decay_params_["eta0"] = eta0_;
        lr_decay_params_["gamma"] = gamma_;
        lr_decay_ = LRDecayRegistry()->Create(lr_policy_, lr_decay_params_);
    }

    void init_optimizer_params() {}

public:
    BaseOptimizer() {};
    BaseOptimizer(const std::vector<FeatureType>& w0,
                  const FeatureType& b0,
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
                  bool verbose = true): w0_(w0), b0_(b0),
            loss_(loss), 
            lr_policy_(lr_policy),
            alpha_(alpha),
            eta0_(eta0),
            tol_(tol),
            gamma_(gamma),
            max_iters_(max_iters), 
            batch_size_(batch_size),
            num_iters_no_change_(num_iters_no_change),
            random_seed_(random_seed),
            shuffle_(shuffle),
            verbose_(verbose) {
        if (random_seed_ == -1) {
            random_state_ = sgdlib::internal::RandomState();
        }
        else {
            random_state_ = sgdlib::internal::RandomState(random_seed_);
        }
        init_loss_params();
        init_lr_params();
    };

    BaseOptimizer(const std::vector<FeatureType>& w0,
                  const FeatureType& b0,
                  std::string loss, 
                  double alpha,
                  double eta0,
                  double tol,
                  std::size_t max_iters, 
                  std::size_t random_seed,
                  bool is_saga = false,
                  bool shuffle = true, 
                  bool verbose = true): w0_(w0), b0_(b0),
            loss_(loss), 
            alpha_(alpha),
            eta0_(eta0),
            tol_(tol),
            max_iters_(max_iters), 
            random_seed_(random_seed),
            is_saga_(is_saga),
            shuffle_(shuffle),
            verbose_(verbose) {
        if (random_seed_ == -1) {
            random_state_ = sgdlib::internal::RandomState();
        }
        else {
            random_state_ = sgdlib::internal::RandomState(random_seed_);
        }
        init_loss_params();
        stepsize_search_param_["alpha"] = alpha;
        stepsize_search_param_["eta0"] = eta0;
    };
    
    ~BaseOptimizer() {};

    virtual void optimize(const std::vector<FeatureType>& X, 
                          const std::vector<LabelType>& y) = 0;

    const std::vector<FeatureType> get_coef() const {
        return w_opt_;
    }

    const FeatureType get_intercept() const {
        return b_opt_;
    }

};

} // namespace sgdlib

#endif // ALGORITHM_BASE_HPP_