//
// Created by Haoda Fu on 2019-06-02.
//

#ifndef MINIXGBOOST_INCLUDE_IO_H_
#define MINIXGBOOST_INCLUDE_IO_H_

#include "../src/matrix.h"

namespace miniXGBoost {

namespace dataIO {
void loadLibSVMData(data::DataSet &data, const std::string &fName);
void loadLibSVMData(data::FeatureMatrix &fMatrix, const std::string &fName);
void loadCSVData(data::DataSet &data, const std::string &fName);
void loadCSVData(data::FeatureMatrix &fMatrix, const std::string &fName);
void simulateData(data::DataSet &data, const std::string &fName);
}  // namespace miniXGBoost::dataIO

namespace modelIO {

}  // namespace miniXGBoost::modelIO
}  // namespace miniXGBoost
#endif //MINIXGBOOST_INCLUDE_IO_H_
