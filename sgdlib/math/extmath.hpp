#ifndef MATH_EXTMATH_HPP_
#define MATH_EXTMATH_HPP_

#include "common/logging.hpp"
#include "common/prereqs.hpp"


namespace sgdlib {
namespace internal {

/**
 * @brief Clip (limit) the values in a vector.
 * 
 * @tparam Type The type of elements in the vector.
 * 
 * @param x vector containing elements to clip.
 * @param min, max minimum and maximum value
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
*/
template<typename Type>
inline void clip(std::vector<Type>& x, Type min, Type max) {
    if (min > max) {
        THROW_INVALID_ERROR("a_min must be less than or equal to a_max.");
    }
    std::transform(std::begin(x), std::end(x), std::begin(x),
        [=] (auto i) { 
            return std::clamp(i, min, max); 
        }
    );
};

/** 
 * @brief Clips the value to the specified range [min, max]
 * 
 * @tparam Type The type of elements for inputs.
 * 
 * @param x The input value to be clipped.
 * @param min, max minimum and maximum value
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
*/
template<typename Type>
inline void clip(Type& x, Type min, Type max) {
    if (min > max) {
        THROW_INVALID_ERROR("a_min must be less than or equal to a_max.");
    }
    x = std::max(min, std::min(x, max));
};

/**
 * @brief check if any element of vector is infinite.
 * 
 * @tparam Type The type of elements in the vector.
 * 
 * @param x vector containing elements to check infinity.
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
*/
template<typename Type>
inline bool isinf(const std::vector<Type>& x) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (std::isinf(x[i])) {
            return true;
        }
    }
    return false;
};

/**
 * @brief check if the given value represents infinity
 * 
 * @tparam Type The type of elements for input.
 * 
 * @param x value to check infinity.
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
 */
template<typename Type>
inline bool isinf(const Type& x) {
    if (std::isinf(x)) {
        return true;
    }
    return false;
};

/**
 * @brief calculate the L2 norm of a vector.
 * 
 * @tparam Type The type of elements in the vector.
 * 
 * @param x a vector of type T.
 * @return The L2 norm of the vector as a custome type.
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
 */
template<typename Type>
inline Type sqnorm2(const std::vector<Type>& x, bool squared) {
    Type norm2 = std::inner_product(x.begin(), 
                                    x.end(), 
                                    x.begin(), 
                                    static_cast<Type>(0.0));
    if (squared) {
        norm2 = std::sqrt(norm2);
    }
    
    return norm2;
};

/** 
 * @brief calculate the L1 norm (Manhattan distance) of a vector.
 * 
 * @tparam Type The type of elements in the vector.
 * 
 * @param x a vector of type T.
 * @return The L1 norm of the vector as a T.
 * 
*/
template<typename Type>
inline Type norm1(const std::vector<Type>& x) {
    Type norm = 0;
    for (const Type& value : x) {
        norm += std::abs(value);
    }
    return norm;
};

/** 
 * @brief Applies a scalar multiplication operation to a vector.
 * 
 * @tparam Type The type of elements in the vector.
 * 
 * @param[in,out] x vector of type T, which will be scaled by the scalar 'c'.
 * @param[in] scalar constant scalar value 
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
*/
template<typename Type>
inline void dot(std::vector<Type>& x, const Type c) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] *= c;
    }
};

/** 
 * @brief Applies a scalar multiplication operation to a vector. 
 * It computes the sum of the products of all elements within 
 * the iterator range from begin to end with a constant c.
 * 
 * @tparam Type The type of elements in the vector.
 * @tparam IterType The type of iterator, with a default of std::vector<Type>::const_iterator.
 * 
 * @param begin The beginning iterator pointing to the first element 
 *      of the vector to calculate the dot product.
 * @param end The ending iterator pointing to the first element following begin.
 * @param c The constant to be multiplied with the elements of the vector.
 * @param out The reference to a vector that stores the result of the dot product.
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
 * @note The use of templates and default iterator types allows the function 
 *       to be used with different types of vectors and iterators.
 * 
*/
template<typename Type, 
         typename IterType = typename std::vector<Type>::const_iterator>
inline void dot(IterType begin, IterType end, 
                const Type c, 
                std::vector<Type>& out) {
    if(std::distance(begin, end) != out.size()){
        THROW_INVALID_ERROR("Output vector size is insufficient.");
    }
    std::transform(begin, end, 
                   out.begin(), 
                   [c](const Type& elem) {
                        return elem * c;
                   });
};

/** 
 * @brief Applies a scalar multiplication operation to a vector. 
 * It computes the sum of the products of all elements within 
 * the iterator range from begin to end with a constant c.
 * 
 * @tparam Type The type of elements in the vector.
 * 
 * @param[in] x vector of type T,
 * @param[in] y vector of type T,
 * @param out The reference to a vector that stores the result of the dot product.
 * 
 * @note The function is marked as inline, which is suitable for small functions 
 *       to reduce the overhead of function calls.
 * 
 */
template <typename Type>
inline void dot(const std::vector<Type>& x, 
                const std::vector<Type>& y, 
                Type& out) {
    if (x.size() != y.size()) {
        THROW_INVALID_ERROR("Vectors must have the same size.");
    }
    out = std::inner_product(x.begin(), x.end(), y.begin(), 0.0);
}

template<typename Type>
void inline add(const std::vector<Type>& x, 
                const std::vector<Type>& y, 
                std::vector<Type>& out) {
    // check
    if (x.size() != y.size()) {
        THROW_INVALID_ERROR("Vectors must have the same size.");
    }

    out.resize(x.size());
    std::transform(x.begin(), x.end(), y.begin(), out.begin(), std::plus<Type>());
}

/**
 * @brief Computes the row-wise norms of a vector.
 *
 * This function calculates the norms of each row in a given vector `x`,
 * The result can be squared or not, based on the `squared` flag.
 *
 * @tparam Type A floating-point or arithmetic type that supports necessary operations.
 * @param x A constant reference to the input vector
 * @param squared A boolean flag that determines whether to compute the squared norms.
 *                - If true, the function computes the squared Euclidean norm: 
 *                  \( \sum_{i=1}^{n} x_i^2 \)
 *                - If false, the function computes the Euclidean norm (non-squared), which
 *                  is the square root of the squared norm.
 * @param out A reference to the output vector where the computed norms will be stored.
 *
 * @note The function assumes that `x` represents a matrix with its size is ncols * nrows.
 *
*/
template<typename Type>
void row_norms(const std::vector<Type>& x, 
               bool squared,
               std::vector<Type>& out) {
    
    std::size_t num_elems = x.size();
    std::size_t nrows = out.size();
    std::size_t ncols = num_elems / nrows;
    std::vector<Type> elem_prod(num_elems);
    
    // compute x * x
    std::transform(x.begin(), x.end(), elem_prod.begin(),
                    [](const Type& value) { 
                        return value * value;
                    });
    
    // compute prefix sum of elem_prod
    std::vector<Type> prefix_sum(num_elems);
    std::partial_sum(elem_prod.begin(), elem_prod.end(), prefix_sum.begin());

    // compute the sum of every nth element
    std::size_t count = 0;
    for (std::size_t i = 0; i < num_elems; i += ncols) {
        std::size_t end = std::min(i + ncols, num_elems);
        if (count == 0) {
            out[count] = prefix_sum[end - 1];
        } else {
            out[count] = prefix_sum[end - 1] - prefix_sum[i - 1];
        }
        count += 1;
    }

    if (!squared) {
        std::transform(out.begin(), out.end(), out.begin(),
                    [](const Type& value) {
                        return std::sqrt(value);
                    });
    }
};


/**
 * @brief Computes the column-wise norms of a vector.
 *
 * This function calculates the norms of columns in a vector, treating it as a 2D matrix.
 * The norms can be either squared or not, based on the 'squared' parameter.
 *
 * @tparam Type The data type of the vector elements.
 * @param[in] x The input vector to compute norms from, treated as a 2D matrix.
 * @param[in] squared If true, computes squared norms; if false, computes regular norms.
 * @param[out] out The output vector to store the computed norms. Its size determines the number of columns.
 *
 * @note The function assumes that the size of 'x' is divisible by the size of 'out'.
 */
template<typename Type>
void col_norms(const std::vector<Type>& x, 
               bool squared,
               std::vector<Type>& out) {
    
    std::size_t num_elems = x.size();
    std::size_t ncols = out.size();
    std::size_t nrows = num_elems / ncols;

    std::size_t j = 0;
    while (j < ncols) {
        Type sum_sq = 0.0;
        for (std::size_t i = 0; i < nrows; ++i) {
            Type val = x[j + i * ncols];
            sum_sq += val * val; 
        }
        out[j] = sum_sq;
        ++j;
    }

    if (!squared) {
        std::transform(out.begin(), out.end(), out.begin(),
                    [](const Type& value) {
                        return std::sqrt(value);
                    });
    }
};


} // namespace internal
} // namespace sgdlib

#endif // MATH_EXTMATH_HPP_