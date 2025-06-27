#ifndef ALGORITHM_BASE_HPP_
#define ALGORITHM_BASE_HPP_

#include "common/constants.hpp"
#include "common/params.hpp"
#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "common/logging.hpp"
#include "common/types.hpp"
#include "core/loss.hpp"
#include "core/lr_decay.hpp"
#include "core/stepsize_search.hpp"
#include "math/random.hpp"
#include "math/math_ops.hpp"
#include "data/continuous_dataset.hpp"

namespace sgdlib {

class Optimizer {
public:
    // public callback type for optimizer
    using CallbackType = std::function<void(const std::vector<sgdlib::FeatureScalarType>&)>;

    // default constructor
    Optimizer() {};

    // base constructor
    Optimizer(const std::vector<sgdlib::FeatureScalarType>& w0,
        std::string loss,
        sgdlib::ScalarType tol,
        std::size_t max_iters,
        bool verbose = true): w0_(w0),
            loss_(loss),
            tol_(tol),
            max_iters_(max_iters),
            verbose_(verbose),
            callback_(nullptr) {
        this->init_loss_params();
    };

    virtual ~Optimizer() = default;

    virtual void optimize(const sgdlib::ArrayDatasetType& dataset) = 0;

    virtual void optimize(const std::vector<sgdlib::FeatureScalarType>& X,
                          const std::vector<sgdlib::LabelScalarType>& y) = 0;

    const std::vector<sgdlib::FeatureScalarType> get_weights() const {
        return w_opt_;
    }

    virtual const sgdlib::FeatureScalarType get_intercept() const {
        return b_opt_;
    }

    void set_callback(CallbackType callback) {
        callback_ = callback;
    }

protected:
    std::vector<sgdlib::FeatureScalarType> w0_;
    sgdlib::FeatureScalarType b0_;

    std::string loss_;
    std::string penalty_;
    std::string lr_policy_;
    std::string search_policy_;

    sgdlib::ScalarType alpha_;
    sgdlib::ScalarType beta_;
    sgdlib::ScalarType delta_;
    sgdlib::ScalarType rho_;
    sgdlib::ScalarType eta0_;
    sgdlib::ScalarType tol_;
    sgdlib::ScalarType gamma_;

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

    sgdlib::LossParamType loss_params_;
    sgdlib::LRDecayParamType lr_decay_params_;
    std::shared_ptr<sgdlib::StepSizeSearchParamType> stepsize_search_params_;

    std::vector<std::size_t> X_data_index_;
    std::vector<sgdlib::FeatureScalarType> loss_history_;
    std::vector<sgdlib::FeatureScalarType> w_opt_;
    sgdlib::FeatureScalarType b_opt_;
    sgdlib::detail::RandomState random_state_;

    std::shared_ptr<sgdlib::detail::LossFunctionType> loss_fn_;
    std::unique_ptr<sgdlib::detail::LRDecay> lr_decay_;

    CallbackType callback_;

    void init_random_state() {
        if (random_seed_ == -1) {
            random_state_ = sgdlib::detail::RandomState();
        }
        else {
            random_state_ = sgdlib::detail::RandomState(random_seed_);
        }
    }

    void init_loss_params() {
        // initialize loss function
        // margin threshold for hinge loss
        loss_params_["threshold"] = 1.0;
        loss_fn_ = sgdlib::detail::LossFunctionRegistry()->Create(loss_, loss_params_);
    }

    void init_lr_params() {
        // initialize learning rate scheduler
        lr_decay_params_["eta0"] = eta0_;
        lr_decay_params_["gamma"] = gamma_;
        lr_decay_ = sgdlib::detail::LRDecayRegistry()->Create(lr_policy_, lr_decay_params_);
    }

};

} // namespace sgdlib

#endif // ALGORITHM_BASE_HPP_
