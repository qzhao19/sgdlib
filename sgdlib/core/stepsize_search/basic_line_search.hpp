#ifndef CORE_STEPSIZE_SEARCH_BASIC_LINE_EARCH_HPP_
#define CORE_STEPSIZE_SEARCH_BASIC_LINE_EARCH_HPP_

#include "base.hpp"

namespace sgdlib {

class BasicLineSearch final: public StepSizeSearch{
public:
    BasicLineSearch(const std::vector<FeatureType>& X, 
                   const std::vector<LabelType>& y,
                   StepSizeSearchParamType stepsize_search_param): StepSizeSearch(X, y, stepsize_search_param) {};
    ~BasicLineSearch() {};

    /** 
     * Compute step size and it is specifically used for the SAG optimizer.
    */
    int search(bool is_saga, double& step_size) override {
        return 0;
    }
};

} // namespace sgdlib

#endif // CORE_STEPSIZE_SEARCH_BASIC_LINE_EARCH_HPP_