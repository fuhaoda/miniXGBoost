
//
// Created by Haoda Fu on 2019-06-06.
//

#ifndef MINIXGBOOST_SRC_GBTREEPREDICTOR_H_
#define MINIXGBOOST_SRC_GBTREEPREDICTOR_H_

#include <vector>
#include "miniXGBoost.h"
#include "matrix.h"


namespace miniXGBoost {
class GBPredictor {
 public:
  GBPredictor(const data::FeatureMatrix &fMatrix, const miniXGBoost::Model &model) :
      fMatrix_{fMatrix},
      model_{model} {};

  // based on feature matrix X, predict the outcome of y
  std::vector<float> predict();

 private:
  const data::FeatureMatrix &fMatrix_;
  const Model &model_;
  void singleTreePrediction(const std::vector<TreeNode> & tree, std::vector<float> & prediction);
};
}

#endif //MINIXGBOOST_SRC_GBTREEPREDICTOR_H_
