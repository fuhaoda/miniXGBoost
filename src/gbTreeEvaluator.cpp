//
// Created by Haoda Fu on 2019-06-06.
//

#include "gbTreeEvaluator.h"
#include "utils.h"
float miniXGBoost::GBEvaluator::getLoss() {
  auto yhat = predict();
  utils::myAssert(yhat.size() == y_.size(), "Vector length doesn't match: predicted values vs "
                                            "observed values");

  float loss{0.0f};
  for (size_t i = 0; i < yhat.size(); ++i) {
    loss += fun_.loss(y_[i], yhat[i]);
  }

  return loss;
}
