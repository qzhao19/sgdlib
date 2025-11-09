#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <thread>
#include <cmath>
#include "sgdlib/data/dataset_manager.hpp"

namespace fs = std::filesystem;

class DatasetManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // create temp file with sample data
        temp_file_ = fs::temp_directory_path() / "test_data";
        std::ofstream file(temp_file_.string());
        file << "1 1:0.5 2:-0.3 3:1.2\n";
        file << "-1 1:0.1 2:-0.5 3:0.8\n";
        file << "1 1:-0.3 2:0.7 3:0.2\n";
        file.close();

        if (fs::file_size(temp_file_) == 0) {
            FAIL() << "Created empty test file";
        }
    }

    void TearDown() override {
        if (fs::exists(temp_file_)) {
            fs::remove(temp_file_);
        }
        sgdlib::DatasetManager<double, int>::get_instance(3).clear();
    }

    fs::path temp_file_;
};

// Basic load test
TEST_F(DatasetManagerTest, BasicLoad) {
    auto& dm = sgdlib::DatasetManager<double, int>::get_instance(3);
    dm.load(temp_file_.string());

    EXPECT_EQ(dm.get_num_samples(), 3);
    EXPECT_EQ(dm.get_num_features(), 3);
    EXPECT_EQ(dm.get_labels().size(), 3);
    EXPECT_EQ(dm.get_features().size(), 9); // 3 samples * 3 features
}

// Column-major storage verification
TEST_F(DatasetManagerTest, ColumnMajorStorage) {
    auto& dm = sgdlib::DatasetManager<double, int>::get_instance(3);
    dm.load(temp_file_.string());

    auto features = dm.get_features();
    // Column-major: [feature0_sample0, feature0_sample1, feature0_sample2,
    //                 feature1_sample0, feature1_sample1, feature1_sample2,
    //                 feature2_sample0, feature2_sample1, feature2_sample2]

    // Feature 0: [0.5, 0.1, -0.3]
    EXPECT_DOUBLE_EQ(features[0], 0.5);
    EXPECT_DOUBLE_EQ(features[1], 0.1);
    EXPECT_DOUBLE_EQ(features[2], -0.3);

    // Feature 1: [-0.3, -0.5, 0.7]
    EXPECT_DOUBLE_EQ(features[3], -0.3);
    EXPECT_DOUBLE_EQ(features[4], -0.5);
    EXPECT_DOUBLE_EQ(features[5], 0.7);

    // Feature 2: [1.2, 0.8, 0.2]
    EXPECT_DOUBLE_EQ(features[6], 1.2);
    EXPECT_DOUBLE_EQ(features[7], 0.8);
    EXPECT_DOUBLE_EQ(features[8], 0.2);
}

// Row sample extraction test
TEST_F(DatasetManagerTest, RowSampleExtraction) {
    auto& dm = sgdlib::DatasetManager<double, int>::get_instance(3);
    dm.load(temp_file_.string());

    // Sample 0: [0.5, -0.3, 1.2]
    auto sample0 = dm.get_row_sample(0);
    EXPECT_EQ(sample0.size(), 3);
    EXPECT_DOUBLE_EQ(sample0[0], 0.5);
    EXPECT_DOUBLE_EQ(sample0[1], -0.3);
    EXPECT_DOUBLE_EQ(sample0[2], 1.2);

    // Sample 1: [0.1, -0.5, 0.8]
    auto sample1 = dm.get_row_sample(1);
    EXPECT_DOUBLE_EQ(sample1[0], 0.1);
    EXPECT_DOUBLE_EQ(sample1[1], -0.5);
    EXPECT_DOUBLE_EQ(sample1[2], 0.8);

    // Sample 2: [-0.3, 0.7, 0.2]
    auto sample2 = dm.get_row_sample(2);
    EXPECT_DOUBLE_EQ(sample2[0], -0.3);
    EXPECT_DOUBLE_EQ(sample2[1], 0.7);
    EXPECT_DOUBLE_EQ(sample2[2], 0.2);
}

// Labels test
TEST_F(DatasetManagerTest, LabelsCorrectness) {
    auto& dm = sgdlib::DatasetManager<double, int>::get_instance(3);
    dm.load(temp_file_.string());

    auto labels = dm.get_labels();
    EXPECT_EQ(labels.size(), 3);
    EXPECT_EQ(labels[0], 1);
    EXPECT_EQ(labels[1], -1);
    EXPECT_EQ(labels[2], 1);
}

// Exception test
TEST_F(DatasetManagerTest, ErrorHandling) {
    auto& dm = sgdlib::DatasetManager<double, int>::get_instance(3);

    // Non-existent file check
    EXPECT_THROW(dm.load("nonexistent_file.txt"), std::runtime_error);

    // Invalid label format test
    std::ofstream bad_file(temp_file_);
    bad_file << "invalid_label 1:0.5 2:0.3\n";
    bad_file.close();
    dm.clear();
    EXPECT_THROW(dm.load(temp_file_.string()), std::runtime_error);
}

// Max samples test
TEST_F(DatasetManagerTest, MaxSamplesLimit) {
    auto& dm = sgdlib::DatasetManager<double, int>::get_instance(3);

    dm.set_max_samples(2);
    dm.load(temp_file_.string());

    EXPECT_EQ(dm.get_num_samples(), 2);
    EXPECT_EQ(dm.get_labels().size(), 2);
    EXPECT_EQ(dm.get_features().size(), 6); // 2 samples * 3 features (column-major)

    auto features = dm.get_features();
    // Only first 2 samples loaded
    // Feature 0: [0.5, 0.1]
    EXPECT_DOUBLE_EQ(features[0], 0.5);
    EXPECT_DOUBLE_EQ(features[1], 0.1);
    // Feature 1: [-0.3, -0.5]
    EXPECT_DOUBLE_EQ(features[2], -0.3);
    EXPECT_DOUBLE_EQ(features[3], -0.5);
    // Feature 2: [1.2, 0.8]
    EXPECT_DOUBLE_EQ(features[4], 1.2);
    EXPECT_DOUBLE_EQ(features[5], 0.8);
}

// Thread safety test
TEST_F(DatasetManagerTest, ThreadSafety) {
    auto& dm = sgdlib::DatasetManager<double, int>::get_instance(3);
    dm.load(temp_file_.string());

    constexpr int kThreads = 4;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&dm, &success_count] {
            try {
                auto labels = dm.get_labels();
                auto features = dm.get_features();
                auto sample = dm.get_row_sample(0);

                // Verify data consistency
                if (!labels.empty() && !features.empty() && !sample.empty()) {
                    success_count++;
                }
            } catch (...) {
                // Catch any exception
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count, kThreads);
}

// Singleton test
TEST_F(DatasetManagerTest, SingletonPattern) {
    auto& dm1 = sgdlib::DatasetManager<double, int>::get_instance(3);
    auto& dm2 = sgdlib::DatasetManager<double, int>::get_instance(3);

    EXPECT_EQ(&dm1, &dm2);
}

// Clear state test
TEST_F(DatasetManagerTest, ClearState) {
    auto& dm = sgdlib::DatasetManager<double, int>::get_instance(3);
    dm.load(temp_file_.string());

    EXPECT_GT(dm.get_num_samples(), 0);
    dm.clear();

    EXPECT_EQ(dm.get_num_samples(), 0);
    EXPECT_TRUE(dm.get_labels().empty());
    EXPECT_TRUE(dm.get_features().empty());
}

// Feature parsing test (scientific notation, inf, etc.)
TEST_F(DatasetManagerTest, FeatureParsing) {
    auto& dm = sgdlib::DatasetManager<double, int>::get_instance(3);

    std::ofstream file(temp_file_.string());
    file << "1 1:0.5 2:-1e-3 3:1.2e+1\n";
    file << "-1 3:inf 2:-inf\n";
    file.close();

    dm.load(temp_file_.string());

    const auto sample0 = dm.get_row_sample(0);
    EXPECT_DOUBLE_EQ(sample0[0], 0.5);
    EXPECT_DOUBLE_EQ(sample0[1], -0.001);
    EXPECT_DOUBLE_EQ(sample0[2], 12.0);

    const auto sample1 = dm.get_row_sample(1);
    EXPECT_DOUBLE_EQ(sample1[0], 0.0); // Default fill
    EXPECT_TRUE(std::isinf(sample1[1]) && sample1[1] < 0); // -inf
    EXPECT_TRUE(std::isinf(sample1[2]) && sample1[2] > 0); // +inf
}

// Out of range index test
TEST_F(DatasetManagerTest, OutOfRangeAccess) {
    auto& dm = sgdlib::DatasetManager<double, int>::get_instance(3);
    dm.load(temp_file_.string());

    EXPECT_THROW(dm.get_row_sample(10), std::out_of_range);
}

