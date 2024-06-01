#ifndef ALGORITHM_BASE_HPP_
#define ALGORITHM_BASE_HPP_

#include "common/prereqs.hpp"
#include "math/random.hpp"

namespace sgdlib {

class BaseOptimizer {
private:
    std::vector<FeatureType> x0_;
    std::string loss_;
    std::string penalty_;
    std::string lr_policy_;

    double alpha_;
    double eta0_;
    double tol_;
    double decay_;

    std::size_t max_iters_;
    std::size_t batch_size_;
    std::size_t num_iters_no_change_;
    std::size_t random_seed_;

    bool fit_intercept_;
    bool multi_class_;
    bool shuffle_;
    bool verbose_;

protected:
    sgdlib::internal::RandomState random_state_;

public:
    BaseOptimizer() {};
    BaseOptimizer(const std::vector<FeatureType>& x0, 
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
                  bool fit_intercept,
                  bool multi_class,
                  bool shuffle = true, 
                  bool verbose = true): x0_(x0), 
            loss_(loss), 
            penalty_(penalty),
            lr_policy_(lr_policy),
            alpha_(alpha),
            eta0_(eta0),
            tol_(tol),
            decay_(decay),
            max_iters_(max_iters), 
            batch_size_(batch_size),
            num_iters_no_change_(num_iters_no_change),
            random_seed_(random_seed),
            fit_intercept_(fit_intercept),
            multi_class_(multi_class),
            shuffle_(shuffle),
            verbose_(verbose) {
        if (random_seed_ == -1) {
            random_state_ = decisiontree::RandomState();
        }
        else {
            random_state_ = decisiontree::RandomState(random_seed_);
        }
    };
    
    ~BaseOptimizer() {};

    virtual void optimize(const std::vector<FeatureType>& X, 
                          const std::vector<LabelType>& y) = 0;

};

} // namespace sgdlib

#endif // ALGORITHM_BASE_HPP_