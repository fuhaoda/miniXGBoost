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

  size_t splitFeatureIndex{0};
  float splitValue{0};
};

//FullTreeNode contains the complete tree node information during model building stage.
struct FullTreeNode:public TreeNode{

};

}  // namespace miniXGBoost

#endif //MINIXGBOOST_SRC_TREENODES_H_
