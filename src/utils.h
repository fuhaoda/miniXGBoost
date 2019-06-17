//
// Created on 2019-06-01.
//

#ifndef MINIXGBOOST_SRC_UTILS_H_
#define MINIXGBOOST_SRC_UTILS_H_

#include <iostream>

namespace miniXGBoost::utils {

inline void error(const std::string &msg) {
  fprintf(stderr, "Error: %s\n", msg.c_str());
  exit(-1);
}

inline void myAssert(bool exp) {
  if (!exp) error("Assert Error!");
}

inline void myAssert(bool exp, const std::string &msg) {
  if (!exp) error(msg);
}

inline void warning(const std::string &msg) {
  fprintf(stderr, "Warning: %s \n", msg.c_str());
}

inline void warning(bool exp, const std::string &msg) {
  if (exp) fprintf(stderr, "Warning: %s \n", msg.c_str());
}
}

#endif //MINIXGBOOST_SRC_UTILS_H_
