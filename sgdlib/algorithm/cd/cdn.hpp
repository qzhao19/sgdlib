#ifndef ALGORITHM_CD_CDN_HPP_
#define ALGORITHM_CD_CDN_HPP_

#include "algorithm/base.hpp"

namespace sgdlib {

class CDN: public BaseOptimizer {
public:
    CDN(const std::vector<FeatValType>& w0,
        std::string loss,
        FloatType alpha,
        FloatType tol,
        std::size_t max_iters,
        std::size_t random_seed,
        bool shuffle = true,
        bool verbose = true): BaseOptimizer(w0,
            loss,
            alpha,
            tol,
            max_iters,
            random_seed,
            shuffle,
            verbose) {};

    ~CDN() = default;

    void optimize(const std::vector<FeatValType>& X,
                  const std::vector<LabelValType>& y) override {
        const std::size_t num_samples = y.size();
        const std::size_t num_features = this->w0_.size();
        const FeatValType inv_num_samples = 1.0 / static_cast<FeatValType>(num_samples);

        




    }

};

}
#endif // ALGORITHM_CD_SCD_HPP_
