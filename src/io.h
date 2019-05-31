#ifndef MINIXGBOOST_IO_H
#define MINIXGBOOST_IO_H

#include <vector>
#include <string>
#include "matrix.h"

void loadSVMData(const std::string &fname,
                 std::vector<std::vector<Entry>> &feature_matrix,
                 std::vector<float> &resp); 

#endif
