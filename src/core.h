//
// Created by Haoda Fu on 2019-06-01.
//

#ifndef MINIXGBOOST_SRC_CORE_H_
#define MINIXGBOOST_SRC_CORE_H_

#include <vector>
#include "miniXGBoost.h"
#include "matrix.h"

namespace miniXGBoost {
class GBMCore {
 public:
  GBMCore(ModelParam &param, LossFunction &func) : param_{param}, func_{func},
  max_nodes_{(1U << (param.max_depth + 1)) - 1} {}

  // Train the model.
  void train(const std::string &train_data);

  // Evaluate the trained model on a testing data set.
  void evaluate();

  // predict results based on feature matrix only.
  void predict(const std::string &test_data, std::vector<float> &pred) const;


 private:

  // Model parameter.
  ModelParam & param_;

  // Loss function.
  LossFunction & func_;

  // Maximum number of tree nodes within each boosting tree.
  size_t max_nodes_;


};
}  // namespace miniXGBoost

#endif //MINIXGBOOST_SRC_CORE_H_
