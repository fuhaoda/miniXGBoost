#ifndef MINIXGBOOST_CORE_H
#define MINIXGBOOST_CORE_H

#include <string> 
#include <vector>
#include "loss_function.h"
#include "tree.h"

// Model parameters
struct ModelParam {
  // Reference: arXiv: 1603.02754v3

  // Number of gradient boosting trees to construct, variable K in Eq. (1) 
  size_t num_round = 10;

  // Parameters of the penalty term in Eq. (2). Additionally, by Eq. (7), gamma
  // is also used as the minimal loss change permitted by a split. 
  float gamma = 1.0;
  float lambda = 1.0; 

  // Maximum depth of each boosting tree. Depth 0 corresponds to the root layer.
  size_t max_depth = 3;

  // Shrinkage weights. Each newly added weights will be shrinked by eta after
  // each step of tree boosting (when a new layer of the tree is added). 
  float eta = 0.3; 

  // Minimum weight permitted in a leaf node.
  float min_weight = 1.0;

  // Path to training data set.
  std::string training_data_path; 
};

// Core gradient boosting class. 
class GBM {
public:
  GBM(ModelParam &param, LossFunction &lfunc): param_{param}, lfunc_{lfunc} { } 

  // Train the model.
  void train();

  // Apply the model on testing data set.
  void apply(const std::string &data_path, std::vector<float> &yhat) const; 

private:
  // Model parameters.
  ModelParam param_;

  // Loss function.
  LossFunction lfunc_; 
  
  // Tree nodes of each boosting tree. Each tree reserves the storage for a
  // complete binary tree. 
  std::vector<TreeNode> tree_;

  // Feature matrix
  SparseMatrix matrix_;

  // Response vector
  std::vector<float> response_;

  // Predicted response of the training samples. 
  std::vector<float> prediction_;
}; 


#endif 

