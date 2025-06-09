#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "sgdlib/data/continuous_dataset.hpp"

class ArrayDatasetTest : public ::testing::Test {
public:
    std::vector<double> X_data;
    std::vector<long> y_data;
    void SetUp() override {
        X_data = {
            5.1, 4.9, 4.7, 4.6, 5. , 5.4, 4.6, 5. , 4.4, 4.9, 5.4, 4.8, 4.8,
            4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5. ,
            5. , 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5. , 5.5, 4.9, 4.4,
            5.1, 5. , 4.5, 4.4, 5. , 5.1, 4.8, 5.1, 4.6, 5.3, 5. , 7. , 6.4,
            6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5. , 5.9, 6. , 6.1, 5.6,
            6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7,
            6. , 5.7, 5.5, 5.5, 5.8, 6. , 5.4, 6. , 6.7, 6.3, 5.6, 5.5, 5.5,
            6.1, 5.8, 5. , 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3,
            6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5,
            7.7, 7.7, 6. , 6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2,
            7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6. , 6.9, 6.7, 6.9, 5.8,
            6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9,
            3.5, 3. , 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3. ,
            3. , 4. , 4.4, 3.9, 3.5, 3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3. ,
            3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 4.1, 4.2, 3.1, 3.2, 3.5, 3.6, 3. ,
            3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3. , 3.8, 3.2, 3.7, 3.3, 3.2, 3.2,
            3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2. , 3. , 2.2, 2.9, 2.9,
            3.1, 3. , 2.7, 2.2, 2.5, 3.2, 2.8, 2.5, 2.8, 2.9, 3. , 2.8, 3. ,
            2.9, 2.6, 2.4, 2.4, 2.7, 2.7, 3. , 3.4, 3.1, 2.3, 3. , 2.5, 2.6,
            3. , 2.6, 2.3, 2.7, 3. , 2.9, 2.9, 2.5, 2.8, 3.3, 2.7, 3. , 2.9,
            3. , 3. , 2.5, 2.9, 2.5, 3.6, 3.2, 2.7, 3. , 2.5, 2.8, 3.2, 3. ,
            3.8, 2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8, 3. , 2.8, 3. ,
            2.8, 3.8, 2.8, 2.8, 2.6, 3. , 3.4, 3.1, 3. , 3.1, 3.1, 3.1, 2.7,
            3.2, 3.3, 3. , 2.5, 3. , 3.4, 3.,
            1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4,
            1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5, 1.7, 1.5, 1. , 1.7, 1.9, 1.6,
            1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.4, 1.3,
            1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7, 4.5,
            4.9, 4. , 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4. , 4.7, 3.6,
            4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4. , 4.9, 4.7, 4.3, 4.4, 4.8, 5. ,
            4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4. , 4.4,
            4.6, 4. , 3.3, 4.2, 4.2, 4.2, 4.3, 3. , 4.1, 6. , 5.1, 5.9, 5.6,
            5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5, 5. , 5.1, 5.3, 5.5,
            6.7, 6.9, 5. , 5.7, 4.9, 6.7, 4.9, 5.7, 6. , 4.8, 4.9, 5.6, 5.8,
            6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.1,
            5.9, 5.7, 5.2, 5. , 5.2, 5.4, 5.1,
            0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1,
            0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2,
            0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2,
            0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.4, 1.5,
            1.5, 1.3, 1.5, 1.3, 1.6, 1. , 1.3, 1.4, 1. , 1.5, 1. , 1.4, 1.3,
            1.4, 1.5, 1. , 1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7,
            1.5, 1. , 1.1, 1. , 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3, 1.2,
            1.4, 1.2, 1. , 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.5, 1.9, 2.1, 1.8,
            2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2. , 1.9, 2.1, 2. , 2.4, 2.3, 1.8,
            2.2, 2.3, 1.5, 2.3, 2. , 2. , 1.8, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6,
            1.9, 2. , 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9,
            2.3, 2.5, 2.3, 1.9, 2. , 2.3, 1.8
        };
        y_data = {
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    }
};


TEST_F(ArrayDatasetTest, ConstructorValidation) {
    EXPECT_EQ(X_data.size(), 600);
    EXPECT_EQ(y_data.size(), 150);

    EXPECT_NO_THROW(sgdlib::detail::ArrayDataset(X_data, y_data, 150, 4));
    EXPECT_THROW(sgdlib::detail::ArrayDataset(X_data, y_data, 151, 4), std::invalid_argument);
};

TEST_F(ArrayDatasetTest, RowAccessWithoutCache) {
    sgdlib::detail::ArrayDataset dataset(X_data, y_data, 150, 4, false);

    std::vector<double> x_row(4);
    long y_row;

    dataset.X_row_data(0, x_row);
    dataset.y_row_data(0, y_row);
    ASSERT_EQ(x_row.size(), 4);
    EXPECT_FLOAT_EQ(x_row[0], 5.1);
    EXPECT_FLOAT_EQ(x_row[1], 3.5);
    EXPECT_FLOAT_EQ(x_row[2], 1.4);
    EXPECT_FLOAT_EQ(x_row[3], 0.2);
    EXPECT_EQ(y_row, -1);

    dataset.X_row_data(1, x_row);
    dataset.y_row_data(1, y_row);
    ASSERT_EQ(x_row.size(), 4);
    EXPECT_FLOAT_EQ(x_row[0], 4.9);
    EXPECT_FLOAT_EQ(x_row[1], 3.0);
    EXPECT_FLOAT_EQ(x_row[2], 1.4);
    EXPECT_FLOAT_EQ(x_row[3], 0.2);
    EXPECT_EQ(y_row, -1);

    EXPECT_EQ(dataset.nrows(), 150);
    EXPECT_EQ(dataset.ncols(), 4);
    EXPECT_FALSE(dataset.is_cache_enabled());

}

TEST_F(ArrayDatasetTest, RowAccessWithCache) {
    sgdlib::detail::ArrayDataset dataset(X_data, y_data, 150, 4);

    std::vector<double> x_row(4);
    long y_row;

    dataset.X_row_data(0, x_row);
    dataset.y_row_data(0, y_row);
    ASSERT_EQ(x_row.size(), 4);
    EXPECT_FLOAT_EQ(x_row[0], 5.1);
    EXPECT_FLOAT_EQ(x_row[1], 3.5);
    EXPECT_FLOAT_EQ(x_row[2], 1.4);
    EXPECT_FLOAT_EQ(x_row[3], 0.2);
    EXPECT_EQ(y_row, -1);

    dataset.X_row_data(1, x_row);
    dataset.y_row_data(1, y_row);
    ASSERT_EQ(x_row.size(), 4);
    EXPECT_FLOAT_EQ(x_row[0], 4.9);
    EXPECT_FLOAT_EQ(x_row[1], 3.0);
    EXPECT_FLOAT_EQ(x_row[2], 1.4);
    EXPECT_FLOAT_EQ(x_row[3], 0.2);
    EXPECT_EQ(y_row, -1);

    EXPECT_EQ(dataset.nrows(), 150);
    EXPECT_EQ(dataset.ncols(), 4);
    EXPECT_TRUE(dataset.is_cache_enabled());
}

TEST_F(ArrayDatasetTest, ColumnAccess) {
    sgdlib::detail::ArrayDataset dataset(X_data, y_data, 150, 4);

    std::vector<double> x_column(150);
    std::vector<long> y_column(150);

    std::vector<double> expect_X = {
        5.1, 4.9, 4.7, 4.6, 5. , 5.4, 4.6, 5. , 4.4, 4.9, 5.4, 4.8, 4.8,
        4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5. ,
        5. , 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5. , 5.5, 4.9, 4.4,
        5.1, 5. , 4.5, 4.4, 5. , 5.1, 4.8, 5.1, 4.6, 5.3, 5. , 7. , 6.4,
        6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5. , 5.9, 6. , 6.1, 5.6,
        6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7,
        6. , 5.7, 5.5, 5.5, 5.8, 6. , 5.4, 6. , 6.7, 6.3, 5.6, 5.5, 5.5,
        6.1, 5.8, 5. , 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3,
        6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5,
        7.7, 7.7, 6. , 6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2,
        7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6. , 6.9, 6.7, 6.9, 5.8,
        6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9
    };

    std::vector<long> expect_y = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };

    dataset.X_column_data(0, x_column);
    dataset.y_column_data(y_column);

    for (std::size_t i = 0; i < 150; ++i) {
        EXPECT_FLOAT_EQ(expect_X[i], x_column[i]);
        EXPECT_FLOAT_EQ(expect_y[i], y_column[i]);
    }
}

TEST_F(ArrayDatasetTest, RowAccessPerformance) {
    std::size_t nrows = 150;
    std::size_t ncols = 4;
    std::size_t num_loops = 1000000;
    std::vector<double> X_data(nrows * ncols, 1.0);
    std::vector<long> y_data(nrows, 0);

    std::vector<double> x_row(4);
    long y_row;
    sgdlib::detail::ArrayDataset dataset_cached(X_data, y_data, nrows, ncols);
    std::size_t index;

    auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < num_loops; ++i) {
        index = num_loops % nrows;
        dataset_cached.X_row_data(index, x_row);
        dataset_cached.y_row_data(index, y_row);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    std::cout << "Cached ArrayDataset execution time: " << elapsed1.count() << " seconds\n";

    sgdlib::detail::ArrayDataset dataset_uncached(X_data, y_data, nrows, ncols, false);
    start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < num_loops; ++i) {
        index = num_loops % nrows;
        dataset_uncached.X_row_data(index, x_row);
        dataset_uncached.y_row_data(index, y_row);
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed2 = end - start;
    std::cout << "Uncached ArrayDataset execution time: " << elapsed1.count() << " seconds\n";
    std::cout << "Speedup: " << elapsed2.count() / elapsed1.count() << "x\n";
}



