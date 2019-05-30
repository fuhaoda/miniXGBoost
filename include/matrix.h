#ifndef MINIXGBOOST_MATRIX_H
#define MINIXGBOOST_MATRIX_H

#include <vector>

// Nonzero entry of a sparse matrix. For CSR format, the index field corresponds
// to the column index of the nonzero entry. For CSC format, the index field
// corresponds to the row index of the nonzero entry. 
struct Entry {
  Entry(size_t i, float v): index{i}, value{v} { }

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
  // data[i] corresponds to the covariates of the ith sample. Only the
  // covariates whose values are nonzero are included. 
  SparseMatrix(const std::vector<std::vector<Entry>> &data,
               bool csc = false, bool sorted = false);
  
  // Return the number of rows.
  size_t nRows() const { return nrows_; }

  // Return the number of columns.
  size_t nCols() const { return ncols_; } 

  // Return the begin of a row/column. 
  const Entry *begin(size_t i, bool column = false) const;

  // Return one entry pass the end of a row/column.
  const Entry *end(size_t i, bool column = false) const;
  
private:
  // Dimensions of the matrix
  size_t nrows_, ncols_; 
  
  // Row offsets and data for CSR format 
  std::vector<size_t> row_ptr_;  
  std::vector<Entry> row_data_;

  // Column offsets and data for CSC format
  std::vector<size_t> col_ptr_; 
  std::vector<Entry> col_data_;
};



#endif 
