#ifndef CORE_LOSS_BASE_HPP_
#define CORE_LOSS_BASE_HPP_

#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "common/registry.hpp"
#include "math/extmath.hpp"

namespace sgdlib {

class LossFunction {
protected:
    LossParamType loss_param_;
public:
    LossFunction(LossParamType loss_param): loss_param_(loss_param) {};
    virtual ~LossFunction() {};

    /* -------------without intercept term------------- */
    /**
     * @brief evaluate the loss function
     * 
     * @param[in] X 1darray of shape (num_samples * num_features), the matrix of input data
     * @param[in] y 1darray of shape (num_samples) 
     * @param[in] w 1darray of shape (num_features) coefficient of the features
     * @param[in] b scalar intercept term
     * @return loss function value
    */
    virtual double evaluate(const std::vector<FeatureType>& X, 
                            const std::vector<LabelType>& y, 
                            const std::vector<FeatureType>& w, 
                            const FeatureType& b) const = 0 ;
    
    /**
     * @brief compute gradient of the loss function
     * 
     * @param[in] X 1darray of shape (num_samples * num_features), the matrix of input data
     * @param[in] y 1darray of shape (num_samples) 
     * @param[in] w 1darray of shape (num_features) coefficient of the features
     * @param[in] b scalar intercept term of the model
     * @param[out] grad_w vector of FeatureType, where the gradient with respect to 
     *      the weights will be stored.
     * @param[out] grad_b FeatureType, where the gradient with respect to the bias
    */
    virtual void gradient(const std::vector<FeatureType>& X, 
                          const std::vector<LabelType>& y, 
                          const std::vector<FeatureType>& w,
                          const FeatureType& b,
                          std::vector<FeatureType>& grad_w, 
                          FeatureType& grad_b) const = 0;
    
    /* -------------compute intercept term------------- */

    /**
     * evaluate the loss value of model
     * 
     * @param[in] X 1darray of shape (num_samples * num_features), the matrix of input data
     * @param[in] y 1darray of shape (num_samples) 
     * @param[in] w 1darray of shape (num_features) coefficient of the features
    */
    virtual double evaluate(const std::vector<FeatureType>& X, 
                            const std::vector<LabelType>& y, 
                            const std::vector<FeatureType>& w) const = 0 ;

    /** 
     * compute gradient of the loss function 
    */
    virtual void gradient(const std::vector<FeatureType>& X, 
                          const std::vector<LabelType>& y, 
                          const std::vector<FeatureType>& w,
                          std::vector<FeatureType>& grad) const = 0;

    // set loss function parameters
    void set_param(const std::string& name, const double& value) {
        loss_param_[name] = value;
    }
};

// Create registries for base loss function
DECLARE_REGISTRY(LossFunctionRegistry, LossFunction, LossParamType);
DEFINE_REGISTRY(LossFunctionRegistry, LossFunction, LossParamType);

} // namespace sgdlib

#endif // CORE_LOSS_BASE_HPP_