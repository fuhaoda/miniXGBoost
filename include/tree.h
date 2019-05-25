//
// Created by Haoda Fu on 2019-05-24.
//

#ifndef MINIXGBOOST_TREE_H
#define MINIXGBOOST_TREE_H

#include <iostream>
#include <vector>

namespace xgboost{
  namespace tree{

    struct TreeNode{
      int parent;
      int leftChild;
      int rightChild;
      float sumGrad;
      float sumHess;

      float weight();
      float gain();
    }; //end of TreeNode Struct definition


    class GBTreeModel{

    private:
      std::vector<std::vector<TreeNode>> GBRegModel;
    };


    class GBSingleTreeGenerator{
    private:
      std::vector<TreeNode> & newGBTree;
      std::vector<int> positions_;
      std::vector<int> qexpand_;

      void initData();
      void initNewNode();
      void findSplit();
      void resetPosition();
      void updateQueueExpand();
    };

  } //namespace tree
} //namespace xgboost
#endif //MINIXGBOOST_TREE_H
