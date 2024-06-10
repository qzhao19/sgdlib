#ifndef CORE_LOSS_HUBER_LOSS_HPP_
#define CORE_LOSS_HUBER_LOSS_HPP_

#include "base.hpp"

namespace sgdlib {

class HuberLoss final: public LossFunction {
public:
    HuberLoss(LossParamType loss_param): LossFunction(loss_param) {};
    ~HuberLoss() {};

    /**
     * evaluate the logistic regression log-likelihood function
     * with the given parameters.
     *   - sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)) / len(y_hat)
     * @param[in] X 1darray of shape (num_samples * num_features), the matrix of input data
     * @param[in] y 1darray of shape (num_samples) 
     * @param[in] w 1darray of shape (num_features) coefficient of the features 
     * 
     * @return loss function value 
    */
    double evaluate(const std::vector<FeatureType>& X, 
                    const std::vector<LabelType>& y, 
                    const std::vector<FeatureType>& w) const override {
        std::size_t num_samples = y.size();
        std::size_t num_features = w.size();  
        double loss = 0.0;
        double X_w, y_hat;
        for (std::size_t i = 0; i < num_samples; ++i) {
            X_w = std::inner_product(X.begin() + (i * num_features), 
                                     X.begin() + ((i + 1) * num_features), 
                                     w.begin(), 0.0);      
            y_hat = sgdlib::internal::sigmoid<FeatureType>(X_w);

            loss -= (static_cast<double>(y[i]) * std::log(y_hat) + 
                    (1.0 - static_cast<double>(y[i])) * std::log(1.0 - y_hat));
        }
        loss /= static_cast<double>(num_samples);
        
        // calculate 
        double reg = std::inner_product(w.begin(), 
                                        w.end(), 
                                        w.begin(), 0.0);
        reg /= static_cast<double>(num_samples);
        
        return loss + reg * loss_param_.at("alpha");
    }

    /**
     * Evaluate the gradient of logistic regression log-likelihood
     * function with the given parameters using the given batch 
     * size from the given point label
     * 
     * @param[in] X 1darray of shape (num_samples * num_features), the matrix of input data
     * @param[in] y 1darray of shape (num_samples) 
     * @param[in] w 1darray of shape (num_features) coefficient of the features 
    */
    void gradient(const std::vector<FeatureType>& X, 
                  const std::vector<LabelType>& y, 
                  const std::vector<FeatureType>& w,
                  std::vector<FeatureType>& grad) const override {
        std::size_t num_samples = y.size();
        std::size_t num_features = w.size();
        double X_w;
        std::vector<FeatureType> y_hat(num_samples, 0.0);

        for (std::size_t i = 0; i < num_samples; ++i) {
            X_w = std::inner_product(X.begin() + (i * num_features), 
                                     X.begin() + ((i + 1) * num_features), 
                                     w.begin(), 0.0);
            y_hat[i] = sgdlib::internal::sigmoid<FeatureType>(X_w);
        }

        FeatureType inner_prod = 0.0;
        std::size_t fx_index = 0;
        while (fx_index < num_features) {
            for (std::size_t i = 0; i < num_samples; ++i) {
                inner_prod += (X[i * num_features + fx_index]) * 
                              (y_hat[i] - static_cast<FeatureType>(y[i]));
            }
            grad[fx_index] = inner_prod / static_cast<FeatureType>(num_samples) + 
                             loss_param_.at("alpha") * 2.0 * w[fx_index];
            ++fx_index;
            inner_prod = 0.0;
        }
    }

};

} // namespace sgdlib

#endif // CORE_LOSS_HUBER_LOSS_HPP_