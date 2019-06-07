//
// Created by Haoda Fu on 2019-06-06.
//

#ifndef MINIXGBOOST_SRC_GBTREEEVALUATOR_H_
#define MINIXGBOOST_SRC_GBTREEEVALUATOR_H_

#include "gbTreePredictor.h"
namespace miniXGBoost {
class GBEvaluator : public GBPredictor {
 public:
  GBEvaluator(const data::DataSet &data, const miniXGBoost::Model &model, const LossFunction &fun)
      : GBPredictor(data.X,
                    model), y_{data.y}, fun_{fun} {}

  // return evaluation loss
  float getLoss();

 private:
  const std::vector<float> &y_;
  const LossFunction &fun_;

};
}

#endif //MINIXGBOOST_SRC_GBTREEEVALUATOR_H_
