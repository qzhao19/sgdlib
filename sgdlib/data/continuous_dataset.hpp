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
    // define unique_ptr to manager row cache
    std::unique_ptr<FeatValType[]> X_row_cache_;
    std::shared_ptr<const FeatValType[]> X_data_ptr_;
    std::shared_ptr<const LabelValType[]> y_data_ptr_;

    std::size_t nrows_, ncols_;
    bool enable_cache_;

    void col2row_cache() {
        #if defined(USE_OPENMP)
        #pragma omp parallel for
        #endif
        for (std::size_t r = 0; r < nrows_; ++r) {
            FeatValType* x_row_ptr = X_row_cache_.get() + r * ncols_;
            for (std::size_t c = 0; c < ncols_; ++c) {
                x_row_ptr[c] = X_data_ptr_[r + c * nrows_];
            }
        }
    };

public:
    // constructor for raw ptr
    ArrayDataset(std::shared_ptr<const FeatValType[]> X_data_ptr,
        std::shared_ptr<const LabelValType[]> y_data_ptr,
        std::size_t nrows,
        std::size_t ncols,
        bool enable_cache = true):
            X_data_ptr_(X_data_ptr),
            y_data_ptr_(y_data_ptr),
            nrows_(nrows),
            ncols_(ncols),
            enable_cache_(enable_cache) {

        if (!X_data_ptr || !y_data_ptr) {
            throw std::invalid_argument("Data pointers cannot be null");
        }

        if (enable_cache_) {
            X_row_cache_ = std::make_unique<FeatValType[]>(nrows * ncols);
            col2row_cache();
        }
    };

    ArrayDataset(const std::vector<FeatValType>& X_data,
        const std::vector<LabelValType>& y_data,
        std::size_t nrows,
        std::size_t ncols,
        bool enable_cache = true) : ArrayDataset(
            std::shared_ptr<const FeatValType[]>(X_data.data(), [](auto ptr){}),
            std::shared_ptr<const LabelValType[]>(y_data.data(), [](auto ptr){}),
            nrows,
            ncols,
            enable_cache) {

        if (X_data.size() < nrows * ncols) {
            THROW_INVALID_ERROR("X_data size is smaller than nrows * ncols");
        }

        if (y_data.size() < nrows) {
            THROW_INVALID_ERROR("y_data size is smaller than nrows");
        }
    };

    ~ArrayDataset() = default;

    void X_row_data(const std::size_t i, std::vector<FeatValType>& row) {
        if (i >= nrows_) {
            THROW_OUT_RANGE_ERROR("Row index out of range");
        }
        if (enable_cache_) {
            std::memcpy(row.data(),
                X_row_cache_.get() + i * ncols_,
                ncols_ * sizeof(FeatValType));
        }
        else {
            // collect data from col-major-order X
            for (std::size_t c = 0; c < ncols_; ++c) {
                row[c] = X_data_ptr_[i + c * nrows_];
            }
        }
    }

    void y_row_data(const std::size_t i, LabelValType& row) {
        if (i >= nrows_) {
            THROW_OUT_RANGE_ERROR("Row index out of range");
        }
        row = y_data_ptr_[i];
    }

    void X_column_data(const std::size_t j, std::vector<FeatValType>& column) {
        if (j >= ncols_) {
            THROW_OUT_RANGE_ERROR("Column index out of range");
        }
        std::memcpy(column.data(),
            X_data_ptr_.get() + j * nrows_,
            nrows_ * sizeof(FeatValType));
    }

    void y_column_data(std::vector<LabelValType>& column) {
        std::memcpy(column.data(), y_data_ptr_.get(), nrows_ * sizeof(LabelValType));
    }

    // read-only
    const FeatValType* X_data_ptr() const { return X_data_ptr_.get(); }
    const LabelValType* y_data_ptr() const { return y_data_ptr_.get(); }

    // read-only row cache
    const FeatValType* row_cache() const { return X_row_cache_.get(); }

    std::size_t nrows() const { return nrows_; }
    std::size_t ncols() const { return ncols_; }
    bool is_cache_enabled() const { return enable_cache_; }

};

}
}

#endif // DATA_CONTINUOUS_DATASET_HPP_
