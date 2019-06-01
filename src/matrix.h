//
// Created by Haoda Fu on 2019-06-01.
//

#ifndef MINIXGBOOST_SRC_MATRIX_H_
#define MINIXGBOOST_SRC_MATRIX_H_

#include <vector>

namespace miniXGBoost::data {

// Nonzero entry of a sparse matrix. For CSR format, the index field corresponds
// to the column index of the nonzero entry. For CSC format, the index field
// corresponds to the row index of the nonzero entry.
struct Entry {
  explicit Entry(size_t i = 0, float v = 0.0) : index{i}, value{v} {}
  size_t index;
  float value;
  bool operator<(const Entry &other) const {
    return value < other.value;
  }
};

// The sparse matrix corresponds to the feature matrix of a training/testing
// data set. Each row of the matrix corresponds to individual sample and the
// columns of the matrix corresponds to the covariates. The dataset are
// typically read by sample. This means, the sparse matrix is saved in CSR
// format by default. However, one needs to examine covariate(s) across all
// samples on a training data set, requiring the matrix to have CSC format
// support.

class SparseMatrix {
 public:
  SparseMatrix() = default;

  // data[i] corresponds to the covariates of the ith sample. Only the
  // covariates whose values are nonzero are included.
  SparseMatrix(const std::vector<std::vector<Entry>> &data, bool csc = false, bool sortColumn =
  false);

  // Return the number of rows.
  size_t nRows() const { return nrows_; }

  // Return the number of columns.
  size_t nCols() const { return ncols_; }

 private:
  // Dimensions of the matrix
  size_t nrows_{0}, ncols_{0};
  // Row offsets and data for CSR format
  std::vector<size_t> row_ptr_{};
  std::vector<Entry> row_data_{};

  // Column offsets and data for CSC format
  std::vector<size_t> col_ptr_{};
  std::vector<Entry> col_data_{};
};
} // namespace miniXGBoost::data

#endif //MINIXGBOOST_SRC_MATRIX_H_
