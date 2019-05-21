//
// Created by Haoda Fu on 2019-05-20.
//

#include <data.h>
#include <fstream>
#include <sstream>
#include "data.h"
#include "base.h"

size_t xgboost::data::SimpleSparseMatrix::addRow(const std::vector<size_t> &findex, const std::vector<float> &fvalue) {
  utils::myassert(findex.size() == fvalue.size());
  for (size_t iter = 0; iter < findex.size(); ++iter) {
    row_data_.emplace_back(findex[iter], fvalue[iter]);
  }
  row_ptr_.push_back(row_ptr_.back() + findex.size());
  return row_ptr_.size() - 2;
}


inline void xgboost::data::SimpleSparseMatrix::clear() {
  row_ptr_.clear();
  row_ptr_.push_back(0);
  row_data_.clear();
  col_ptr_.clear();
  col_data_.clear();
  y_.clear();
}

void xgboost::data::SimpleSparseMatrix::loadLibSVM(const std::string &dataFileName) {

  std::ifstream dFile;
  dFile.open(dataFileName);
  utils::myassert(!dFile.fail(), "Cannot open data file!");
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
          y_.push_back(stof(temp));
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


