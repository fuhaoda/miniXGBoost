//
// Created by Haoda Fu on 2019-06-01.
//
#include "config.h"

miniXGBoost::ModelParam miniXGBoost::configFileParser(const std::string &config_file) {
  miniXGBoost::ModelParam param{};
  miniXGBoost::ConfigParse confParser(config_file);
  confParser.parse();
  confParser.assignParameters(param);
  return param;
}


void miniXGBoost::ConfigParse::parse() {
  pairKeyValue_.clear();

  char delimiter = '=';
  char comment = '#';
  std::string line{};
  std::string name{};
  std::string value{};

  while (!fi_.eof()) {
    //read a line of configure file
    getline(fi_, line);
    // anything beyond # is comment
    line = line.substr(0, line.find(comment));
    // find the = sign
    size_t delimiterPos = line.find(delimiter);
    // anything before = is the name
    name = line.substr(0, delimiterPos);
    // after this = is the value
    value = line.substr(delimiterPos + 1,
                        line.length() - delimiterPos - 1);
    //skip a line if # at beginning or there is no value or no name.
    if (line.empty() || name.empty() || value.empty())
      continue;
    //clean the string
    cleanString(name);
    cleanString(value);
    pairKeyValue_.emplace_back(name, value);
  }

}
void miniXGBoost::ConfigParse::cleanString(std::string &str) {
  size_t firstIndx = str.find_first_of(allowableChar_);
  size_t lastIndx = str.find_last_of(allowableChar_);
  str = str.substr(firstIndx,
                   lastIndx - firstIndx
                       + 1); //this line can be more efficient, but keep as is for simplicity.
}
void miniXGBoost::ConfigParse::assignParameters(miniXGBoost::ModelParam &param) {

  for(auto const & item:pairKeyValue_){
    auto const & key= item.first;
    auto const & value = item.second;
    if (key == "nTrees") {
      param.nTrees = std::stoul(value);
    } else if (key == "reg_nodes") {
      param.reg_nodes = std::stof(value);
    } else if (key == "reg_weights") {
      param.reg_weights = std::stof(value);
    } else if (key == "max_depth") {
      param.max_depth = std::stoul(value);
    } else if (key == "shrinkage") {
      param.shrinkage = std::stof(value);
    } else if (key == "min_weight") {
      param.min_weight = std::stof(value);
    } else {
      utils::warning("Some keys in configuration file cannot be parsed!");
    }
  }
}
