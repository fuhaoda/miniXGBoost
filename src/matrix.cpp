#include <algorithm>
#include "matrix.h"

SparseMatrix::SparseMatrix(const std::vector<std::vector<Entry>> &data,
                           bool csc, bool sorted) {
  // Get the number of rows. 
  nrows_ = data.size(); 

  // Resize row pointers and zero out. 
  row_ptr_.resize(nrows_ + 1);
  std::fill(row_ptr_.begin(), row_ptr_.end(), 0); 

  // Count number of nonzero entries in each row. 
  for (size_t i = 1; i <= nrows_; ++i) 
    row_ptr_[i] = data[i].size();

  // Set up row pointers. 
  for (size_t i = 2; i <= nrows_; ++i)
    row_ptr_[i] += row_ptr_[i - 1];

  // Resize row data
  row_data_.resize(row_ptr_[nrows_]);

  // Copy the input data into contiguous memory space.
  for (size_t i = 0; i < nrows_; ++i)
    std::copy(data[i].begin(), data[i].end(), &row_data_[row_ptr_[i]]); 

  if (csc == false)
    return;

  // Add CSC format support.

  // Find the maximum column index. 
  ncols_ = 0; 
  for (const Entry &e : row_data_) 
    ncols_ = std::max(ncols_, e.index);

  // Resize column pointers and zero out. 
  col_ptr_.resize(ncols_ + 1);
  std::fill(col_ptr_.begin(), col_ptr_.end(), 0);

  // Scan the row data again to count nonzero entries in each column.
  for (const Entry &e: row_data_)
    col_ptr_[e.index + 1]++;

  // Set up column pointers.
  for (size_t i = 2; i < ncols_; ++i)
    col_ptr_[i] += col_ptr_[i - 1];

  // Resize column data and populate.
  col_data_.resize(col_ptr_[ncols_]);

  std::vector<size_t> filled(ncols_, 0);
  for (const std::vector<Entry> &row : data) {
    for (const Entry &e: row) {
      size_t col = e.index;
      size_t pos = col_ptr_[col] + filled[col];
      col_data_[pos] = e;
      filled[col]++; 
    }
  }
      
  if (sorted == false)
    return;

  // Within each column, sort the entries to faciliate future processing. 
  for (size_t i = 0; i < ncols_; ++i)
    std::sort(&col_data_[col_ptr_[i]], &col_data_[col_ptr_[i + 1]]);   
}

const Entry *SparseMatrix::begin(size_t i, bool column) const {
  return (column ? &col_data_[col_ptr_[i]] : &row_data_[row_ptr_[i]]);
}


const Entry *SparseMatrix::end(size_t i, bool column) const {
  return (column ? &col_data_[col_ptr_[i + 1]] : &row_data_[row_ptr_[i + 1]]); 
}
