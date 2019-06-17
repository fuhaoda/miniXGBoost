//
// Created by Haoda Fu on 2019-06-01.
//

#include "matrix.h"
#include <algorithm>

void miniXGBoost::data::SparseMatrix::clear() {
  row_ptr_.clear();
  row_ptr_.push_back(0);
  row_data_.clear();
  col_ptr_.clear();
  col_data_.clear();
}

size_t miniXGBoost::data::SparseMatrix::addOneRow(const std::vector<miniXGBoost::data::Entry> &
    oneRow) {
  row_data_.insert(row_data_.end(),oneRow.begin(),oneRow.end());
  row_ptr_.push_back(row_ptr_.back() + oneRow.size());
  nrows_ = row_ptr_.size() - 1;
  return row_ptr_.size() - 2;
}


void miniXGBoost::data::SparseMatrix::translateToCSCFormat() {

  ncols_ = 0;
  for (const Entry &e : row_data_)
    ncols_ = std::max(ncols_, e.index);
  ++ncols_;

  col_data_.clear();
  col_ptr_.clear();

  col_ptr_.resize(ncols_ + 1, 0);
  col_data_.resize(row_data_.size());

  //count how many elements in each column
  for (const Entry &e: row_data_)
    ++col_ptr_[e.index + 1];

  size_t rightBount = 0;
  for (auto iter = col_ptr_.begin() + 1; iter != col_ptr_.end(); ++iter) {
    size_t rlen = *iter;
    *iter = rightBount;
    rightBount += rlen;
  }


  //go through data again to push items in
  for(size_t i = 0; i < nrows_; ++i){
    auto end=cend(i);
    for(auto iter = cbegin(i); iter!=end; ++iter){
      col_data_[col_ptr_[iter->index + 1]++] = Entry(i, iter->value);
    }
  }

  // sort columns
  for (unsigned i = 0; i < ncols_; ++i) {
    std::sort(&col_data_[col_ptr_[i]], &col_data_[col_ptr_[i + 1]]);
  }
}

const miniXGBoost::data::Entry *miniXGBoost::data::SparseMatrix::cbegin(size_t i,
                                                                        bool column) const {
  return (column ? &col_data_[col_ptr_[i]] : &row_data_[row_ptr_[i]]);
}
const miniXGBoost::data::Entry *miniXGBoost::data::SparseMatrix::cend(size_t i, bool column) const {
  return (column ? &col_data_[col_ptr_[i + 1]] : &row_data_[row_ptr_[i + 1]]);
}

void miniXGBoost::data::DataSet::clear() {
  y.clear();
  X.clear();
}
