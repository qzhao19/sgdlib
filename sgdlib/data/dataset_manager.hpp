#ifndef DATA_DATASET_MANAGER_HPP_
#define DATA_DATASET_MANAGER_HPP_

#include "common/consts.hpp"
#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "common/logging.hpp"

namespace sgdlib {

template <typename FeatType, typename LabelType>
class DatasetManager {
public:
    static DatasetManager& get_instance(std::size_t num_features = 0) {
        static DatasetManager instance(num_features);
        return instance;
    }

    void set_max_samples(std::size_t max_samples) {
        std::lock_guard lock(mutex_);
        max_samples_ = max_samples;
        if (loaded_ && max_samples_ > 0) {
            trim_datasets();
        }
    }

    void load(std::string_view filepath) {
        std::lock_guard lock(mutex_);
        if (loaded_) return;

        if (num_features_ == 0) {
            THROW_LOGIC_ERROR("num_features must be positive");
        }

        std::ifstream file(filepath.data());
        if (!file) {
            THROW_RUNTIME_ERROR("Failed to open file: ", filepath.data());
        }

        clear_unsafe(); // Clear existing data

        std::string line;
        std::size_t samples_loaded = 0;
        features_.reserve(max_samples_ > 0 ? max_samples_ * num_features_ : 1024 * num_features_);

        while (samples_loaded < max_samples_ && std::getline(file, line)) {
            std::istringstream iss(line);
            LabelType label;
            if (!(iss >> label)) {
                THROW_RUNTIME_ERROR("Invalid label in: ", line);
            }

            std::vector<FeatType> sample(num_features_, 0.0);
            parse_features(iss, sample);
            
            labels_.push_back(std::move(label));
            features_.insert(features_.end(), sample.begin(), sample.end());
            ++samples_loaded;
        }

        if (labels_.empty()) {
            THROW_RUNTIME_ERROR("Empty dataset");
        }

        loaded_ = true;
    }

    // get labels vector (thread-safe)
    std::vector<LabelType> get_labels() const {
        std::lock_guard lock(mutex_);
        return labels_;
    }

    // get features vector (thread-safe)
    std::vector<FeatType> get_features() const {
        std::lock_guard lock(mutex_);
        return features_;
    }

    // get single sample (thread-safe)
    std::vector<FeatType> get_row_sample(std::size_t index) const {
        std::lock_guard lock(mutex_);
        validate_index(index);
        return {features_.begin() + index * num_features_,
                features_.begin() + (index + 1) * num_features_};
    }

    std::size_t get_num_samples() const {
        std::lock_guard lock(mutex_);
        return labels_.size();
    }

    std::size_t get_num_features() const {
        std::lock_guard lock(mutex_);
        return num_features_;
    }

    std::size_t get_max_samples() const {
        std::lock_guard lock(mutex_);
        return max_samples_;
    }

    void clear() {
        std::lock_guard lock(mutex_);
        clear_unsafe();
    }

    DatasetManager(const DatasetManager&) = delete;
    DatasetManager& operator=(const DatasetManager&) = delete;

private:
    explicit DatasetManager(std::size_t num_features) 
        : num_features_(num_features) {
        if (num_features == 0) {
            THROW_INVALID_ERROR("num_features cannot be zero");
        }
    }

    ~DatasetManager() = default;

    void parse_features(std::istringstream& iss, std::vector<FeatType>& sample) {
        std::string token;
        while (iss >> token) {
            const auto colon_pos = token.find(':');
            if (colon_pos == std::string::npos) {
                THROW_RUNTIME_ERROR("Missing ':' in feature: ", token);
            }

            try {
                const std::size_t dim = std::stoul(token.substr(0, colon_pos));
                const FeatType value = [&]{
                    if constexpr (std::is_same_v<FeatType, float>) {
                        return std::stof(token.substr(colon_pos + 1));
                    } 
                    else if constexpr (std::is_same_v<FeatType, double>) {
                        return std::stod(token.substr(colon_pos + 1));
                    } 
                    else {
                        return static_cast<FeatType>(std::stod(token.substr(colon_pos + 1)));
                    }
                }();

                if (dim == 0 || dim > num_features_) {
                    THROW_OUT_RANGE_ERROR("Feature dim ", std::to_string(dim), 
                        " out of range [1,", std::to_string(num_features_), "]");
                }
                sample[dim - 1] = value;
            } catch (const std::exception& e) {
                THROW_RUNTIME_ERROR("Failed to parse feature '", token, "': ", e.what());
            }
        }
    }

    void validate_index(std::size_t index) const {
        if (index >= labels_.size()) {
            THROW_OUT_RANGE_ERROR(
                "Index ", std::to_string(index), 
                " >= sample count ", std::to_string(labels_.size()));
        }
    }

    void trim_datasets() {
        if (labels_.size() > max_samples_) {
            labels_.resize(max_samples_);
            features_.resize(max_samples_ * num_features_);
        }
    }

    void clear_unsafe() {
        labels_.clear();
        features_.clear();
        loaded_ = false;
    }

    std::vector<LabelType> labels_;
    std::vector<FeatType> features_;
    const std::size_t num_features_; 
    bool loaded_ = false;
    std::size_t max_samples_ = 0;
    mutable std::mutex mutex_;
};
    
} // namespace sgdlib
#endif /*DATA_DATASET_MANAGER_HPP_ */ 
