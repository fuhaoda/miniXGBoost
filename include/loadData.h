//
// Created by Haoda Fu on 2019-05-20.
//

#ifndef MINIXGBOOST_LOADDATA_H
#define MINIXGBOOST_LOADDATA_H

#include "data.h"
#include "base.h"
#include <iostream>

namespace xgboost{
  namespace data{

    enum class InternalDataFormat {SimpleSparseMatrix, DenseMatrix};

    class LoadData {
    public:
      // constructor: we can overload the constructor for other internal data format.
      LoadData(SimpleSparseMatrix spm, std::vector<float> y):spm_(spm),y_(y){ internalDataFormat= InternalDataFormat::SimpleSparseMatrix;};

      void loadLibSVM(const std::string & dataFileName);
      void loadCSV(const std::string & dataFileName);
    private:
      SimpleSparseMatrix & spm_;
      std::vector<float> & y_;
      InternalDataFormat internalDataFormat;
    };
  } //namespace data
} //namespace xgboost


#endif //MINIXGBOOST_LOADDATA_H
