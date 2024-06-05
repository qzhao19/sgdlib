#ifndef ALGORITHM_BASE_HPP_
#define ALGORITHM_BASE_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "math/random.hpp"

namespace sgdlib {

class BaseOptimizer {
protected:
    std::vector<FeatureType> x0_;
    std::string loss_;
    std::string penalty_;
    std::string lr_policy_;

    double alpha_;
    double l1_ratio_;
    double eta0_;
    double tol_;
    double decay_;

    std::size_t max_iters_;
    std::size_t batch_size_;
    std::size_t num_iters_no_change_;
    std::size_t random_seed_;

    bool shuffle_;
    bool verbose_;

    LossParamType loss_params_;
    std::vector<FeatureType> x_opt_;
    sgdlib::internal::RandomState random_state_;

public:
    BaseOptimizer() {};
    BaseOptimizer(const std::vector<FeatureType>& x0, 
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
                  bool verbose = true): x0_(x0), 
            loss_(loss), 
            lr_policy_(lr_policy),
            alpha_(alpha),
            eta0_(eta0),
            tol_(tol),
            decay_(decay),
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
        loss_params_["alpha"] = alpha;
    };
    
    ~BaseOptimizer() {};

    virtual void optimize(const std::vector<FeatureType>& X, 
                          const std::vector<LabelType>& y) = 0;

    const std::vector<FeatureType> get_coef() const {
        return x_opt_;
    }

};

} // namespace sgdlib

#endif // ALGORITHM_BASE_HPP_