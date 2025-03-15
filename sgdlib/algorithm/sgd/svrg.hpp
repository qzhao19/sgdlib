#ifndef ALGORITHM_SGD_SVRG_HPP_
#define ALGORITHM_SGD_SVRG_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

/**
 * @file svrg.hpp
 * 
 * @class SVRG
 * 
 * @brief Implements the Stochastic Variance Reduced Gradient (SVRG) optimization algorithm.
 *
*/
class SVRG: public BaseOptimizer {
public:
    /**
     * @brief Constructor for the SVRG optimizer.
     *
     * Initializes the SVRG optimizer with the given parameters and passes them to the
     * base class `BaseOptimizer`.
     *
     * @param w0 Initial weight vector for the model.
     * @param b0 Initial bias term for the model.
     * @param loss The loss function to be minimized.
     * @param lr_policy The learning rate policy (e.g., constant, adaptive).
     * @param alpha L2 regularization parameter.
     * @param eta0 Initial learning rate.
     * @param tol Tolerance for convergence.
     * @param gamma Decay factor for the learning rate (used in some learning rate policies).
     * @param max_iters Maximum number of iterations for optimization.
     * @param batch_size Size of the mini-batch used for gradient computation.
     * @param num_iters_no_change Number of iterations with no improvement to wait before stopping.
     * @param random_seed Seed for the random number generator.
     * @param shuffle If true, shuffles the data before each epoch (default: true).
     * @param verbose If true, enables logging of optimization progress (default: true).
     * 
     * @note This constructor calls the constructor of the base class `BaseOptimizer` to 
     *       complete the initialization of the optimizer.
     * @see BaseOptimizer
     */
    SVRG(const std::vector<FeatValType>& w0, 
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
        bool verbose = true): BaseOptimizer(w0, b0,
            loss, lr_policy, 
            alpha, eta0, 
            tol, 
            gamma,
            max_iters, 
            batch_size, 
            num_iters_no_change,
            random_seed,
            shuffle, 
            verbose) {};

    /**
     * @brief Destructor for the SGD optimizer.
     *
     * Default destructor.
     */
    ~SVRG() = default;

    void optimize(const std::vector<FeatValType>& X, 
        const std::vector<LabelValType>& y) override {

        std::size_t num_samples = y.size();
        std::size_t num_features = this->w0_.size();
        std::size_t step_per_iter = num_samples / this->batch_size_;


    }
};

}

#endif // ALGORITHM_SGD_SVRG_HPP_