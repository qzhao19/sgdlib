#ifndef CORE_LOSS_HUBER_LOSS_HPP_
#define CORE_LOSS_HUBER_LOSS_HPP_

#include "base.hpp"

namespace sgdlib {

/** 
 * @file huber_loss.hpp
 * 
 * @brief Huber loss function
*/
class HuberLoss final: public LossFunction {
public:
    LogLoss(LossParamType loss_param): LossFunction(loss_param) {};
    ~LogLoss() {};

    // with intercept term
    double evaluate(const std::vector<FeatureType>& X, 
                    const std::vector<LabelType>& y, 
                    const std::vector<FeatureType>& w, 
                    const FeatureType& b) const override {
    
        std::size_t num_samples = y.size();
        std::size_t num_features = w.size();  
        double loss = 0.0;
        double X_w, y_hat;
        for (std::size_t i = 0; i < num_samples; ++i) {
            X_w = std::inner_product(X.begin() + (i * num_features), 
                                     X.begin() + ((i + 1) * num_features), 
                                     w.begin(), 0.0);
            X_w *= loss_param_.at("wscale");
            X_w += b; 
            y_hat = sgdlib::internal::sigmoid<FeatureType>(X_w);

            loss -= (static_cast<double>(y[i]) * std::log(y_hat) + 
                    (1.0 - static_cast<double>(y[i])) * std::log(1.0 - y_hat));
        }
        loss /= static_cast<double>(num_samples);
        
        // calculate regulization term
        double reg = std::inner_product(w.begin(), 
                                        w.end(), 
                                        w.begin(), 0.0);
        reg /= static_cast<double>(num_samples);
        return loss + reg * loss_param_.at("alpha");
    }

    void gradient(const std::vector<FeatureType>& X, 
                  const std::vector<LabelType>& y, 
                  const std::vector<FeatureType>& w,
                  const FeatureType& b,
                  std::vector<FeatureType>& grad_w, 
                  FeatureType& grad_b) const override {
    
        std::size_t num_samples = y.size();
        std::size_t num_features = w.size();
        double X_w;
        std::vector<FeatureType> y_hat(num_samples, 0.0);

        for (std::size_t i = 0; i < num_samples; ++i) {
            X_w = std::inner_product(X.begin() + (i * num_features), 
                                     X.begin() + ((i + 1) * num_features), 
                                     w.begin(), 0.0);
            X_w *= loss_param_.at("wscale");
            X_w += b;
            y_hat[i] = sgdlib::internal::sigmoid<FeatureType>(X_w);
            grad_b += (y_hat[i] - static_cast<FeatureType>(y[i]));
        }
        grad_b /= static_cast<FeatureType>(num_samples);

        FeatureType inner_prod = 0.0;
        std::size_t feature_index = 0;
        while (feature_index < num_features) {
            for (std::size_t i = 0; i < num_samples; ++i) {
                inner_prod += (X[i * num_features + feature_index]) * 
                              (y_hat[i] - static_cast<FeatureType>(y[i]));
            }
            grad_w[feature_index] = inner_prod / static_cast<FeatureType>(num_samples) + 
                                    loss_param_.at("alpha") * 2.0 * w[feature_index];
            ++feature_index;
            inner_prod = 0.0;
        }
    }

    // without intercept term
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
            X_w *= loss_param_.at("wscale");
            // X_w += b; 
            y_hat = sgdlib::internal::sigmoid<FeatureType>(X_w);

            loss -= (static_cast<double>(y[i]) * std::log(y_hat) + 
                    (1.0 - static_cast<double>(y[i])) * std::log(1.0 - y_hat));
        }
        loss /= static_cast<double>(num_samples);
        
        // calculate regulization term
        double reg = std::inner_product(w.begin(), 
                                        w.end(), 
                                        w.begin(), 0.0);
        reg /= static_cast<double>(num_samples);
        return loss + reg * loss_param_.at("alpha");
    }

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
            X_w *= loss_param_.at("wscale");
            y_hat[i] = sgdlib::internal::sigmoid<FeatureType>(X_w);
        }

        FeatureType inner_prod = 0.0;
        std::size_t feature_index = 0;
        while (feature_index < num_features) {
            for (std::size_t i = 0; i < num_samples; ++i) {
                inner_prod += (X[i * num_features + feature_index]) * 
                              (y_hat[i] - static_cast<FeatureType>(y[i]));
            }
            grad[feature_index] = inner_prod / static_cast<FeatureType>(num_samples) + 
                                  loss_param_.at("alpha") * 2.0 * w[feature_index];
            ++feature_index;
            inner_prod = 0.0;
        }
    }
};


} // namespace sgdlib

#endif // CORE_LOSS_HUBER_LOSS_HPP_