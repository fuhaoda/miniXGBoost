//
// Created by Haoda Fu on 2019-06-02.
//

#ifndef MINIXGBOOST_SRC_CONFIG_H_
#define MINIXGBOOST_SRC_CONFIG_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <miniXGBoost.h>
#include "utils.h"

namespace miniXGBoost {
class ConfigParse {
 public:
  explicit ConfigParse(const std::string &cfgFileName) {
    fi_.open(cfgFileName);
    utils::myAssert(!fi_.fail(), "Cannot open configuration file!");
  }

  ~ConfigParse() {
    fi_.close();
  }

  void parse();
  void assignParameters(miniXGBoost::ModelParam & param);

 private:
  std::ifstream fi_;
  std::vector<std::pair<std::string, std::string>> pairKeyValue_;
  std::string allowableChar_ = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_./-";
  void cleanString(std::string &str);
};

} // namespace miniXGBoost
#endif //MINIXGBOOST_SRC_CONFIG_H_
