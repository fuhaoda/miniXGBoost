//
// Created by Haoda Fu on 2019-05-20.
//

#include <data.h>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "data.h"
#include "base.h"

size_t xgboost::data::SimpleSparseMatrix::addRow(const std::vector<size_t> &findex, const std::vector<float> &fvalue) {
  utils::myAssert(findex.size() == fvalue.size());
  for (size_t iter = 0; iter < findex.size(); ++iter) {
    row_data_.emplace_back(findex[iter], fvalue[iter]);
  }
  row_ptr_.push_back(row_ptr_.back() + findex.size());
  return row_ptr_.size() - 2;
}


void xgboost::data::SimpleSparseMatrix::clear() {
  row_ptr_.clear();
  row_ptr_.push_back(0);
  row_data_.clear();
  col_ptr_.clear();
  col_data_.clear();
  y_.clear();
  colAccess = false;
}

void xgboost::data::SimpleSparseMatrix::loadLibSVM(const std::string &dataFileName) {

  std::ifstream dFile;
  dFile.open(dataFileName);
  utils::myAssert(!dFile.fail(), "Cannot open data file!");
  std::string line{};
  size_t pos{};
  std::string temp{};
  std::vector<size_t> findex{};
  std::vector<float> fvalue{};
  clear();

  while (!dFile.eof()) {
    findex.clear();
    fvalue.clear();
    getline(dFile, line); //read a line from data file
    std::stringstream ssLine(line);

    bool label = true;
    while (!ssLine.eof()) {
      getline(ssLine, temp, ' ');
      if (!temp.empty()) {
        if (label) {
          y_.emplace_back(stof(temp));
          label = false;
        } else {
          pos = temp.find_first_of(':');
          findex.emplace_back(stoul(temp.substr(0, pos)));
          fvalue.emplace_back(stof(temp.substr(pos + 1, temp.size() - pos - 1)));
        }
      }
    }
    addRow(findex, fvalue);
  }
  dFile.close();
}

size_t xgboost::data::SimpleSparseMatrix::numOfRow() const {
  return row_ptr_.size()-1;
}

class xgboost::data::SimpleSparseMatrix::RowIter{
public:
  RowIter(const xgboost::data::Entry * dprt, const xgboost::data::Entry * end):dprt_(dprt), end_(end){}
  bool next(){
    if(dprt_==end_) return false;
    else{
      ++dprt_;
      return true;
    }
  }
  size_t findex() const{
    return dprt_->findex;
  }
  float fvalue() const{
    return dprt_->fvalue;
  }

public:
  const xgboost::data::Entry * dprt_, * end_;
};


class xgboost::data::SimpleSparseMatrix::ColIter : public xgboost::data::SimpleSparseMatrix::RowIter{
  ColIter(const xgboost::data::Entry *dprt, const xgboost::data::Entry * end):RowIter(dprt,end){}
};

size_t xgboost::data::SimpleSparseMatrix::numOfEntry() const {
  return row_data_.size();
}

xgboost::data::SimpleSparseMatrix::RowIter xgboost::data::SimpleSparseMatrix::getARow(size_t rowIndex) const {
  utils::myAssert( rowIndex < numOfRow(), "row id exceed bound" );
  return RowIter( &row_data_[ row_ptr_[rowIndex] ] - 1, &row_data_[ row_ptr_[rowIndex+1] ] - 1 );
}

void xgboost::data::SimpleSparseMatrix::translateToCSCFormat() {
  if(colAccess){
    utils::warning("Matrix has been translated to CSC format! Translation did not perform");
    return;
  }


  auto entryWithMaxCol = std::max_element(row_data_.begin(),row_data_.end(), xgboost::data::Entry::cmp_findex);
  col_data_.clear();
  col_ptr_.clear();
  col_ptr_.resize(entryWithMaxCol->findex+2,0);
  col_data_.resize(numOfEntry());

  //count how many elements in each column
  for( size_t i = 0; i < numOfRow(); i ++ ){
    for( RowIter it = getARow(i); it.next(); ){
      col_ptr_[it.findex()+1]+=1;
    }
  }

  size_t rightBount = 0;
  for(auto iter = col_ptr_.begin()+1;iter!=col_ptr_.end();++iter){
    size_t rlen = *iter;
    *iter = rightBount;
    rightBount += rlen;
  }

  //go through data again to push items in
  for( size_t i = 0; i < numOfRow(); i ++ ){
    for( RowIter it = getARow(i); it.next(); ){
      col_data_[col_ptr_[it.findex()+1]++]=Entry(i,it.fvalue());
    }
  }
  colAccess=true;

  // sort columns
  for( unsigned i = 0; i < numOfCol(); i ++ ){
    std::sort( &col_data_[ col_ptr_[ i ] ], &col_data_[ col_ptr_[ i+1 ] ], Entry::cmp_fvalue);
  }
}

size_t xgboost::data::SimpleSparseMatrix::numOfCol() const {
  utils::myAssert(colAccess,"No CSC format matrix. Convert the matrix to CSC format first");
  return col_ptr_.size() - 1;
}
