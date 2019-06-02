//
// Created by Haoda Fu on 2019-06-02.
//

#include <fstream>
#include <sstream>
#include "io.h"
#include "matrix.h"
#include "utils.h"
namespace miniXGBoost::dataIO{


void loadLibSVMData(data::DataSet &data, const std::string &fName) {
  std::ifstream dFile;
  dFile.open(fName);
  utils::myAssert(!dFile.fail(), "Cannot open data file!");
  std::string line{};
  size_t pos{};
  std::string temp{};
  data.clear();


  while (!dFile.eof()) {
    // read a line from data file
    getline(dFile, line);
    // if this line only contains space, skip this line
    if(line.find_first_not_of(' ') == std::string::npos) continue;

    std::vector<data::Entry> oneRow{};
    std::stringstream ssLine(line);
    bool response = true;
    while (!ssLine.eof()) {
      getline(ssLine, temp, ' ');
      if (!temp.empty()) {
        if (response) {
          data.y.emplace_back(stof(temp));
          response = false;
        } else {
          pos = temp.find_first_of(':');
          //emplace the index value pair into vector of Entry
          oneRow.emplace_back(stoul(temp.substr(0, pos)), stof(temp.substr(pos + 1, temp.size() -
          pos - 1)));
        }
      }
    }
    data.X.addOneRow(oneRow);
  }
  dFile.close();

  if(data.csc){
    data.X.translateToCSCFormat();
  } else {
    return;
  }

}

void dataIO::loadLibSVMData(data::FeatureMatrix &fMatrix, const std::string &fName) {

}

}  // namespace miniXGBoost::dataIO