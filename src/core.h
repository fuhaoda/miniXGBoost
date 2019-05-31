#ifndef MINIXGBOOST_CORE_H
#define MINIXGBOOST_CORE_H

#include <vector> 
#include "gbm.h"

class GBMCore {
public:
  GBMCore(ModelParam &param, LossFunction &func) : param_{param}, func_{func} { }

  // Train the model.
  void train(const std::string &train_data);

  // Apply the trained model to testing data set.
  void apply(const std::string &test_data, std::vector<float> &pred) const;

private:
  // Model parameter.
  ModelParam param_;

  // Loss function.
  LossFunction func_;

  // Tree nodes of each boosting tree. Each tree reserves the storage for a
  // complete binary tree.
  std::vector<TreeNode> tree_;

  // Feature matrix of the training data set.
  SparseMatrix matrix_;

  // Response vector of the training data set.
  std::vector<float> resp_;

  // Predicted response of the training data set.
  std::vector<float> pred_; 
};

// Factory of models. 
extern std::vector<GBMCore> __model__; 

#endif 

