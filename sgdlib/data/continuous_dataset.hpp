#ifndef DATA_CONTINUOUS_DATASET_HPP_
#define DATA_CONTINUOUS_DATASET_HPP_

#include "common/constants.hpp"
#include "common/prereqs.hpp"
#include "common/predefs.hpp"
#include "common/logging.hpp"


namespace sgdlib {
namespace detail {

class ArrayDataset {
private:
    struct RowData {
        std::vector<FeatValType> features;
        LabelValType label;
        RowData(std::size_t ncols): features(ncols) {};
    };

    struct ColnumData {
        std::vector<FeatValType> features;
        std::vector<LabelValType> labels;
        ColnumData(std::size_t nrows): features(nrows),
            labels(nrows) {};
    };

    FeatValType* X_row_cache_;
    const FeatValType* X_data_ptr_;
    const LabelValType* y_data_ptr_;
    std::size_t nrows_, ncols_;
    bool enable_cache_;

    void transpose_col2row_cache() {
        #if defined(USE_OPENMP)
        #pragma omp parallel for
        #endif
        for (std::size_t r = 0; r < nrows_; ++r) {
            FeatValType* x_row_ptr = X_row_cache_ + r * ncols_;
            for (std::size_t c = 0; c < ncols_; ++c) {
                x_row_ptr[c] = X_data_ptr_[r + c * nrows_];
            }
        }
    };

public:
    using RowDataType = RowData;
    using ColnumDataType = ColnumData;

    ArrayDataset(const std::vector<FeatValType>& X_data,
        const std::vector<LabelValType>& y_data,
        std::size_t nrows,
        std::size_t ncols,
        bool enable_cache = true): X_data_ptr_(X_data.data()),
            y_data_ptr_(y_data.data()),
            nrows_(nrows),
            ncols_(ncols),
            enable_cache_(enable_cache),
            X_row_cache_(nullptr) {

        if (X_data.size() < nrows * ncols) {
            THROW_INVALID_ERROR("X_data size is smaller than nrows * ncols");
        }

        if (y_data.size() < nrows) {
            THROW_INVALID_ERROR("y_data size is smaller than nrows");
        }
        if (enable_cache_) {
            X_row_cache_ = new FeatValType[nrows * ncols];
            transpose_col2row_cache();
        }
    };

    ~ArrayDataset() {
        if (enable_cache_) {
            delete[] X_row_cache_;
        }

    }

    void get_row_data(std::size_t i, RowDataType& row) {
        if (i >= nrows_) {
            THROW_OUT_RANGE_ERROR("Row index out of range");
        }

        row.label = y_data_ptr_[i];
        if (enable_cache_) {
            std::memcpy(row.features.data(),
                        X_row_cache_ + i * ncols_,
                        ncols_ * sizeof(FeatValType));
        }
        else {
            // collect data from col-major-order X
            for (std::size_t c = 0; c < ncols_; ++c) {
                row.features[c] = X_data_ptr_[i + c * nrows_];
            }
        }
    }

    void get_col_data(std::size_t j, ColnumDataType& col) {
        if (j >= ncols_) {
            THROW_OUT_RANGE_ERROR("Column index out of range");
        }

        std::memcpy(col.features.data(), X_data_ptr_ + j * nrows_, nrows_ * sizeof(FeatValType));
        std::memcpy(col.labels.data(), y_data_ptr_, nrows_ * sizeof(LabelValType));
    }

    std::size_t nrows() const { return nrows_; }
    std::size_t ncols() const { return ncols_; }
    bool is_cache_enabled() const { return enable_cache_; }

};

}
}

#endif // DATA_CONTINUOUS_DATASET_HPP_
