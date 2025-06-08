#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "sgdlib/data/continuous_dataset.hpp"

TEST(ArrayDatasetTest, ConstructorValidation) {
    std::vector<double> X_data(6, 1.0f);
    std::vector<long> y_data(2, 0);

    EXPECT_NO_THROW(sgdlib::detail::ArrayDataset(X_data, y_data, 2, 3));
    EXPECT_THROW(sgdlib::detail::ArrayDataset(X_data, y_data, 3, 3), std::invalid_argument);

    std::vector<long> small_y(1, 0);
    EXPECT_THROW(sgdlib::detail::ArrayDataset(X_data, small_y, 2, 3), std::invalid_argument);
}

TEST(ArrayDatasetTest, RowAccessWithoutCache) {
    std::vector<double> X_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<long> y_data = {10, 20};

    sgdlib::detail::ArrayDataset dataset(X_data, y_data, 2, 3, false);

    auto row0 = dataset.get_row_data(0);
    ASSERT_EQ(row0.features.size(), 3);
    EXPECT_FLOAT_EQ(row0.features[0], 1.0f);
    EXPECT_FLOAT_EQ(row0.features[1], 3.0f);
    EXPECT_FLOAT_EQ(row0.features[2], 5.0f);
    EXPECT_EQ(row0.label, 10);

    auto row1 = dataset.get_row_data(1);
    ASSERT_EQ(row1.features.size(), 3);
    EXPECT_FLOAT_EQ(row1.features[0], 2.0f);
    EXPECT_FLOAT_EQ(row1.features[1], 4.0f);
    EXPECT_FLOAT_EQ(row1.features[2], 6.0f);
    EXPECT_EQ(row1.label, 20);

    EXPECT_THROW(dataset.get_row_data(2), std::out_of_range);
}

TEST(ArrayDatasetTest, RowAccessWithCache) {
    std::vector<double> X_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<long> y_data = {10, 20};

    sgdlib::detail::ArrayDataset dataset(X_data, y_data, 2, 3, true);
    EXPECT_TRUE(dataset.is_cache_enabled());

    auto row0 = dataset.get_row_data(0);
    ASSERT_EQ(row0.features.size(), 3);
    EXPECT_FLOAT_EQ(row0.features[0], 1.0f);
    EXPECT_FLOAT_EQ(row0.features[1], 3.0f);
    EXPECT_FLOAT_EQ(row0.features[2], 5.0f);
    EXPECT_EQ(row0.label, 10);

    auto row1 = dataset.get_row_data(1);
    ASSERT_EQ(row1.features.size(), 3);
    EXPECT_FLOAT_EQ(row1.features[0], 2.0f);
    EXPECT_FLOAT_EQ(row1.features[1], 4.0f);
    EXPECT_FLOAT_EQ(row1.features[2], 6.0f);
    EXPECT_EQ(row1.label, 20);
}

TEST(ArrayDatasetTest, ColumnAccess) {
    std::vector<double> X_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<long> y_data = {10, 20};

    sgdlib::detail::ArrayDataset dataset(X_data, y_data, 2, 3);

    auto col0 = dataset.get_col_data(0);
    ASSERT_EQ(col0.features.size(), 2);
    ASSERT_EQ(col0.labels.size(), 2);
    EXPECT_FLOAT_EQ(col0.features[0], 1.0f);
    EXPECT_FLOAT_EQ(col0.features[1], 2.0f);
    EXPECT_EQ(col0.labels[0], 10);
    EXPECT_EQ(col0.labels[1], 20);

    auto col1 = dataset.get_col_data(1);
    ASSERT_EQ(col1.features.size(), 2);
    ASSERT_EQ(col1.labels.size(), 2);
    EXPECT_FLOAT_EQ(col1.features[0], 3.0f);
    EXPECT_FLOAT_EQ(col1.features[1], 4.0f);
    EXPECT_EQ(col1.labels[0], 10);
    EXPECT_EQ(col1.labels[1], 20);

    auto col2 = dataset.get_col_data(2);
    ASSERT_EQ(col2.features.size(), 2);
    ASSERT_EQ(col2.labels.size(), 2);
    EXPECT_FLOAT_EQ(col2.features[0], 5.0f);
    EXPECT_FLOAT_EQ(col2.features[1], 6.0f);
    EXPECT_EQ(col2.labels[0], 10);
    EXPECT_EQ(col2.labels[1], 20);

    EXPECT_THROW(dataset.get_col_data(3), std::out_of_range);
}

TEST(ArrayDatasetTest, LargeDatasetPerformance) {
    constexpr size_t nrows = 100000;
    constexpr size_t ncols = 25;

    std::vector<double> X_data(nrows * ncols, 1.0f);
    std::vector<long> y_data(nrows, 0);

    sgdlib::detail::ArrayDataset no_cache(X_data, y_data, nrows, ncols, false);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < nrows; i += nrows/100) {
        auto row = no_cache.get_row_data(i);
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    std::cout << "No cache row access: " << duration.count() << " ms\n";

    sgdlib::detail::ArrayDataset with_cache(X_data, y_data, nrows, ncols, true);

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < nrows; i += nrows/100) {
        auto row = with_cache.get_row_data(i);
    }
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    std::cout << "With cache row access: " << duration.count() << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    for (size_t j = 0; j < ncols; j++) {
        auto col = with_cache.get_col_data(j);
    }
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    std::cout << "Column access: " << duration.count() << " ms\n";
}

TEST(ArrayDatasetTest, CacheMemoryManagement) {
    std::vector<double> X_data(6, 1.0f);
    std::vector<long> y_data(2, 0);

    {
        sgdlib::detail::ArrayDataset dataset(X_data, y_data, 2, 3, true);
        EXPECT_TRUE(dataset.is_cache_enabled());

        auto row = dataset.get_row_data(0);
        EXPECT_FLOAT_EQ(row.features[0], 1.0f);
    }
}

