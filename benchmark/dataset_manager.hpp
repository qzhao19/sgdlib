#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <mutex>
#include <stdexcept>

class DatasetManager {
public:
    static DatasetManager& getInstance() {
        static DatasetManager instance;
        return instance;
    }

    void load(const std::string& filepath) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (loaded_) return;

        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        labels_.clear();
        features_.clear();

        std::string line;
        size_t num_features = 0;

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            
            int label;
            if (!(iss >> label)) {
                throw std::runtime_error("Invalid label format in file: " + filepath);
            }
            labels_.push_back(label);

            std::vector<float> row;
            float value;
            while (iss >> value) {
                row.push_back(value);
            }

            if (num_features == 0) {
                num_features = row.size();
            } else if (row.size() != num_features) {
                throw std::runtime_error("Inconsistent feature dimensions in file: " + filepath);
            }

            features_.push_back(std::move(row));
        }

        if (labels_.empty()) {
            throw std::runtime_error("Empty dataset in file: " + filepath);
        }

        loaded_ = true;
    }

    const std::vector<int>& labels() const { return labels_; }
    const std::vector<std::vector<float>>& features() const { return features_; }

    DatasetManager(const DatasetManager&) = delete;
    DatasetManager& operator=(const DatasetManager&) = delete;

private:
    DatasetManager() = default;
    ~DatasetManager() = default;

    std::vector<int> labels_;
    std::vector<std::vector<float>> features_;
    bool loaded_ = false;
    std::mutex mutex_;
};