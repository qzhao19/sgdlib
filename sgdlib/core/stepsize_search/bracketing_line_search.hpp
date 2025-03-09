#ifndef CORE_STEPSIZE_SEARCH_BRACKETING_LINE_SEARCH_HPP_
#define CORE_STEPSIZE_SEARCH_BRACKETING_LINE_SEARCH_HPP_

#include "base.hpp"

namespace sgdlib {

template <typename LossFuncType>
class BracketingLineSearch final: public StepSizeSearch<LossFuncType> {
protected:
    std::size_t num_samples_;
    std::size_t num_features_;

public:
    BracketingLineSearch(const std::vector<FeatValType>& X, 
                         const std::vector<LabelValType>& y,
                         const std::shared_ptr<LossFuncType>& loss_fn,
                            StepSizeSearchParamType* stepsize_search_params): StepSizeSearch<LossFuncType>(
                                X, y, 
                                loss_fn, 
                                stepsize_search_params) {
        num_samples_ = y.size();
        
    };
    ~BracketingLineSearch() = default;

    int search(const std::vector<FeatValType>& xp, 
        const std::vector<FeatValType>& gp, 
        std::vector<FeatValType>& x,
        std::vector<FeatValType>& g, 
        std::vector<FeatValType>& d,
        FeatValType& fx,
        FloatType& stepsize) {
        
        return 0; 
    }

};

} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_BRACKETING_LINE_SEARCH_HPP_