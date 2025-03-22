#ifndef ALGORITHM_BASE_HPP_
#define ALGORITHM_BASE_HPP_

#include "common/consts.hpp"
#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "common/logging.hpp"
#include "core/loss.hpp"
#include "core/lr_decay.hpp"
#include "core/stepsize_search.hpp"
#include "math/random.hpp"
#include "math/extmath.hpp"

namespace sgdlib {

class BaseOptimizer {
protected:
    std::vector<FeatValType> w0_;
    FeatValType b0_;
    
    std::string loss_;
    std::string penalty_;
    std::string lr_policy_;
    std::string search_policy_;

    FloatType alpha_;
    FloatType beta_;
    FloatType delta_;
    FloatType rho_;
    FloatType eta0_;
    FloatType tol_;
    FloatType gamma_;
        
    std::size_t past_;
    std::size_t max_iters_;
    std::size_t mem_size_;
    std::size_t batch_size_;
    std::size_t num_iters_no_change_;
    std::size_t num_inner_;
    std::size_t random_seed_;

    bool is_saga_;
    bool shuffle_;
    bool verbose_;

    LossParamType loss_params_;
    LRDecayParamType lr_decay_params_;
    StepSizeSearchParamType* stepsize_search_params_;

    std::vector<FeatValType> w_opt_;
    FeatValType b_opt_;
    sgdlib::internal::RandomState random_state_;

    std::shared_ptr<sgdlib::LossFunction> loss_fn_;
    std::unique_ptr<sgdlib::LRDecay> lr_decay_;

    void init_random_state() {
        if (random_seed_ == -1) {
            random_state_ = sgdlib::internal::RandomState();
        }
        else {
            random_state_ = sgdlib::internal::RandomState(random_seed_);
        }
    }

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

public:
    BaseOptimizer() {};

    // constructor for SGD optimizer
    BaseOptimizer(const std::vector<FeatValType>& w0,
                  const FeatValType& b0,
                  std::string loss, 
                  std::string lr_policy,
                  FloatType alpha,
                  FloatType eta0,
                  FloatType tol,
                  FloatType gamma,
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
        init_random_state();
        init_loss_params();
        init_lr_params();
    };

    // constructor for SAG/SAGA optimizer
    BaseOptimizer(const std::vector<FeatValType>& w0,
                  const FeatValType& b0,
                  std::string loss, 
                  std::string search_policy,
                  FloatType alpha,
                  FloatType eta0,
                  FloatType tol,
                  std::size_t max_iters, 
                  std::size_t random_seed,
                  bool is_saga = false,
                  bool shuffle = true, 
                  bool verbose = true): w0_(w0), b0_(b0),
            loss_(loss), 
            search_policy_(search_policy),
            alpha_(alpha),
            eta0_(eta0),
            tol_(tol),
            max_iters_(max_iters), 
            random_seed_(random_seed),
            is_saga_(is_saga),
            shuffle_(shuffle),
            verbose_(verbose) {
        init_random_state();
        init_loss_params();
        // initialize stepsize search params;
        stepsize_search_params_ = &DEFAULT_STEPSIZE_SEARCH_PARAMS;
        stepsize_search_params_->alpha = alpha_;
        stepsize_search_params_->eta0 = eta0_;
        stepsize_search_params_->max_searches = 10;
        stepsize_search_params_->max_iters = 20;
    };

    // constructor for SCD optimizer
    BaseOptimizer(const std::vector<FeatValType>& w0,
                  std::string loss,
                  FloatType alpha,
                  FloatType tol,
                  std::size_t max_iters, 
                  std::size_t random_seed,
                  bool shuffle = true, 
                  bool verbose = true): w0_(w0),
            loss_(loss),
            alpha_(alpha),
            tol_(tol),
            max_iters_(max_iters), 
            random_seed_(random_seed),
            shuffle_(shuffle),
            verbose_(verbose) {
        init_random_state();
        init_loss_params();

        if (loss_ == "LogLoss") {
            rho_ = 0.25;
        }
        else {
            rho_ = 1.0;
        }
    };

    // constructor for LBFGS optimizer
    BaseOptimizer(const std::vector<FeatValType>& w0, 
                  std::string loss,
                  std::string search_policy,
                  FloatType delta,
                  FloatType tol,
                  std::size_t max_iters, 
                  std::size_t mem_size,
                  std::size_t past,
                  StepSizeSearchParamType* stepsize_search_params,
                  bool verbose = true): w0_(w0), 
            loss_(loss), 
            search_policy_(search_policy),
            delta_(delta),
            tol_(tol),
            max_iters_(max_iters),
            mem_size_(mem_size),
            past_(past),
            stepsize_search_params_(stepsize_search_params),
            verbose_(verbose) {
        init_loss_params();
        if (stepsize_search_params_ == nullptr) {
            stepsize_search_params_ = &DEFAULT_STEPSIZE_SEARCH_PARAMS;
            PRINT_RUNTIME_INFO(1, "Use default search parameters.");
        }
    };

    // constructor for SVRG optimizer
    BaseOptimizer(const std::vector<FeatValType>& w0,
                std::string loss, 
                std::string lr_policy,
                FloatType alpha,
                FloatType eta0,
                FloatType tol,
                FloatType gamma,
                std::size_t max_iters, 
                std::size_t num_inner,
                std::size_t random_seed,
                bool shuffle = true, 
                bool verbose = true): w0_(w0),
            loss_(loss), 
            lr_policy_(lr_policy),
            alpha_(alpha),
            eta0_(eta0),
            tol_(tol),
            gamma_(gamma),
            max_iters_(max_iters), 
            num_inner_(num_inner),
            random_seed_(random_seed),
            shuffle_(shuffle),
            verbose_(verbose) {
        init_random_state();
        init_loss_params();
        init_lr_params();
            };

    virtual ~BaseOptimizer() = default;

    virtual void optimize(const std::vector<FeatValType>& X, 
                          const std::vector<LabelValType>& y) = 0;

    const std::vector<FeatValType> get_coef() const {
        return w_opt_;
    }

    virtual const FeatValType get_intercept() const {
        return b_opt_;
    }

};

} // namespace sgdlib

#endif // ALGORITHM_BASE_HPP_