#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <thread>
#include "sgdlib/data/dataset_manager.hpp"

namespace fs = std::filesystem;

class DatasetManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // create temp file with sample data
        temp_file_ = fs::temp_directory_path() / "test_data";
        std::ofstream file(temp_file_.string());
        file << "1 1:0.5 2:-0.3 3:1.2\n";
        file << "-1 1:0.1 2:-0.3 3:0.8\n";
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
        sgdlib::DatasetManager<float, int>::get_instance(3).clear();
    }

    fs::path temp_file_;
};
    
// exception test
TEST_F(DatasetManagerTest, ErrorHandling) {
    auto& dm = sgdlib::DatasetManager<float, int>::get_instance(3);
    
    // non-existent file check
    EXPECT_THROW(dm.load("nonexistent_file"), std::runtime_error);
    
    // invalid label format test
    std::ofstream bad_file(temp_file_);
    bad_file << "invalid_label a:b c:d\n";
    bad_file.close();
    EXPECT_THROW(dm.load(temp_file_.string()), std::runtime_error);
}

// max_samples test
TEST_F(DatasetManagerTest, MaxSamplesLimit) {
    auto& dm = sgdlib::DatasetManager<float, int>::get_instance(3);
    
    dm.set_max_samples(2);
    dm.load(temp_file_.string());
    
    EXPECT_EQ(dm.get_num_samples(), 2);
    EXPECT_EQ(dm.get_labels().size(), 2);
    EXPECT_EQ(dm.get_features().size(), 6); // 2 samples * 3 features
}

// thread safty test
TEST_F(DatasetManagerTest, ThreadSafety) {
    auto& dm = sgdlib::DatasetManager<float, int>::get_instance(3);
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
                
                if (!labels.empty() && !features.empty() && !sample.empty()) {
                    success_count++;
                }
            } catch (...) {
                // catch any exception
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_EQ(success_count, kThreads);
}

// singleton test
TEST_F(DatasetManagerTest, SingletonPattern) {
    auto& dm1 = sgdlib::DatasetManager<float, int>::get_instance(3);
    auto& dm2 = sgdlib::DatasetManager<float, int>::get_instance(3);
    
    EXPECT_EQ(&dm1, &dm2);
    
    // testing instance independence with different template parameters
    auto& dm_float = sgdlib::DatasetManager<float, int>::get_instance(3);
    auto& dm_double = sgdlib::DatasetManager<double, int>::get_instance(3);
    EXPECT_NE((void*)&dm_float, (void*)&dm_double);
}

// cleanup test
TEST_F(DatasetManagerTest, ClearState) {
    auto& dm = sgdlib::DatasetManager<float, int>::get_instance(3);
    dm.load(temp_file_.string());
    
    EXPECT_TRUE(dm.get_num_samples() > 0);
    dm.clear();
    
    EXPECT_EQ(dm.get_num_samples(), 0);
    EXPECT_TRUE(dm.get_labels().empty());
    EXPECT_TRUE(dm.get_features().empty());
}

TEST_F(DatasetManagerTest, FeatureParsing) {
    auto& dm = sgdlib::DatasetManager<float, int>::get_instance(3);
    
    std::ofstream file(temp_file_.string());
    file << "1 1:0.5 2:-1e-3 3:1.2e+1\n";
    file << "-1 3:inf 2:-inf\n";
    file.close();
    
    dm.load(temp_file_.string());
    
    const auto sample0 = dm.get_row_sample(0);
    EXPECT_FLOAT_EQ(sample0[0], 0.5f);
    EXPECT_FLOAT_EQ(sample0[1], -0.001f);
    EXPECT_FLOAT_EQ(sample0[2], 12.0f);
    
    const auto sample1 = dm.get_row_sample(1);
    EXPECT_TRUE(std::isinf(sample1[2]));
    EXPECT_TRUE(std::isinf(sample1[1]));
    EXPECT_EQ(sample1[0], 0.0f); // default fill 0
}
