//
// Created by Haoda Fu on 2019-05-20.
//

#include "data.h"
#include "base.h"

size_t xgboost::data::SimpleSparseMatrix::addRow(const std::vector<size_t> &findex, const std::vector<float> &fvalue) {
  utils::myassert(findex.size()==fvalue.size());
  for(size_t iter=0; iter < findex.size();++iter){

    std::cout << findex[iter] << std::endl;
    std::cout << fvalue[iter] << std::endl;

    row_data_.emplace_back(Entry(findex[iter],fvalue[iter]));
  }
  row_ptr_.push_back(row_ptr_.back()+findex.size());
  return row_ptr_.size()-2;
}
