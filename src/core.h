#ifndef MINIXGBOOST_CORE_H
#define MINIXGBOOST_CORE_H

#include <vector>
#include "gbm.h"
#include "tree.h"
#include "matrix.h"

class GBMCore {
public:
  GBMCore(ModelParam &param, LossFunction &func) : param_{param}, func_{func} {
    max_nodes_ = (1 << (param.max_depth + 1)) - 1; 
  }

  // Train the model.
  void train(const std::string &train_data);

  // Apply the trained model to testing data set.
  void apply(const std::string &test_data, std::vector<float> &pred) const; 

private:
  // Model parameter.
  ModelParam param_;

  // Loss function.
  LossFunction func_;

  // Maximum number of tree nodes within each boosting tree.
  size_t max_nodes_; 
  
  // Tree nodes of each boosting tree. Each tree reserves the storage for a
  // complete binary tree.
  std::vector<TreeNode> tree_;

  // Feature matrix of the training data set.
  SparseMatrix matrix_;

  // Response vector of the training data set.
  std::vector<float> resp_;

  // Interal working buffers, holding the current predictions of the
  // training data set, the gradient and hessian of the loss function
  // at each sample point. 
  std::vector<float> pred_, grad_, hess_;

  // Internal working buffer, holding the locations of each sample on
  // each boosting tree being constructed. Location -1 means the
  // sample has falled into a leaf node of the tree and can be skipped
  // for future consideration.
  std::vector<int> pos_; 

  // Compute the gradient and hessian of the loss function at each
  // sample using the current prediction value. Accumulate the sum of
  // the gradient and hessian. 
  void computeGradientHessian(float &sum_grad, float &sum_hess);

  // Build the gradient boosting tree.
  void createGBTree(size_t tid, float sum_grad, float sum_hess);

  // Find the split index/value to further split each newly created
  // leaf nodes. Split index -1 means the node will remain leaf and
  // all associated samples will be skipped for further consideration
  // in the tree. 
  void findSplit(); 
  
};

// Factory of models. 
extern std::vector<GBMCore> __model__; 

#endif 

