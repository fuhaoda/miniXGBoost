//
// Created by Haoda Fu on 2019-05-20.
//

#include <fstream>
#include "loadData.h"
#include <sstream>

void xgboost::data::LoadData::loadLibSVM(const std::string & dataFileName) {

  std::ifstream dFile;
  dFile.open(dataFileName);
  utils::myassert(!dFile.fail(),"Cannot open data file!");

  std::string line{};
  size_t pos{};
  std::string temp{};
  std::vector<size_t > findex{};
  std::vector<float> fvalue{};

  switch (internalDataFormat){
    case InternalDataFormat::SimpleSparseMatrix:
      spm_.clear();
      y_.clear();

      while(!dFile.eof() ){

        findex.clear();
        fvalue.clear();
        getline(dFile, line); //read a line from data file
        std::stringstream ssLine(line);

        bool label = true;
        while(!ssLine.eof()){
          getline(ssLine,temp,' ');
          if(!temp.empty()) {
            if(label){
              y_.push_back(stof(temp));
              label = false;
            } else {
              pos = temp.find_first_of(':');
              findex.emplace_back(stoul(temp.substr(0,pos)));
              fvalue.emplace_back(stof(temp.substr(pos+1, temp.size()-pos-1)));
            }
          }
        }
        spm_.addRow(findex,fvalue);
      }
      break;
    case InternalDataFormat::DenseMatrix:
      break;
  }
  dFile.close();
}
