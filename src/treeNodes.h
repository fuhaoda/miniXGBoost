//
// Created by Haoda Fu on 2019-06-02.
//

#ifndef MINIXGBOOST_SRC_TREENODES_H_
#define MINIXGBOOST_SRC_TREENODES_H_

#include <iostream>

namespace miniXGBoost{

//TreeNode is the minimal information that we need to save for a model
struct TreeNode {
  // Index of the parent, left and right child nodes.
  int parent{-1};
  int lChild{-1};
  int rChild{-1};

  //weight and gain
  float weight{0.0f};
  float gain{0.0f};

  // Default branch for missing value
  bool missing_goto_right{true};

  int splitFeatureIndex{-1};
  float splitValue{0};
};

//FullTreeNode contains the complete tree node information during model building stage.
struct FullTreeNode:public TreeNode{
float sum_grad{0};
float sum_hess{0};

// Sum of the gradient/hessian associated with the future child during split
  // option enumeration.
  float child_grad{0.0}, child_hess{0.0};

  // Last feature value examined. Used to set the split value.
  float last_value{0.0};

  // Best score found when enumerating split options.
  float best_score{-1e10};

  // Reset stat
  void reset();

  // Update the best score.
  void update(size_t index, float grad, float hess, float fvalue, float eps,
              float thres, float lambda, float gamma, bool goto_right);

  void update(size_t index, float delta, float thres, float lambda,
              float gamma, bool goto_right);

  // Return the weight of the tree node.
  float weight(float lambda) { return -sum_grad / (sum_hess + lambda); }


};

}  // namespace miniXGBoost

#endif //MINIXGBOOST_SRC_TREENODES_H_
