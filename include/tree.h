//
// Created by Haoda Fu on 2019-05-24.
//

#ifndef MINIXGBOOST_TREE_H
#define MINIXGBOOST_TREE_H

#include <iostream>
#include <vector>
#include "data.h"
#include "base.h"
#include "lossFunction.h"

namespace xgboost{
  namespace tree{

    struct TreeNode{
      int parent{-1};
      int leftChild{-1};
      int rightChild{-1};
      float sumGrad{0};
      float sumHess{0};
      float weight{0};
      float gain{0};
      void calWeight(float lambda);
      void calGain(float lambda);
    }; //end of TreeNode Struct definition


    class GBTreeModel{
    public:
      void training(const data::SimpleSparseMatrix & traingData, const parameters::ModelParam & param);
      void testing(const data::SimpleSparseMatrix & testingData);
      std::vector<float> predicting(const data::SimpleSparseMatrix & featureData); //no response of y

    private:
      std::vector<std::vector<TreeNode>> GBRegModel;
      void updateGradHess(std::vector<detail::GradientPair> & gpair, const std::vector<float> & y, const std::vector<float> & tempPred, lossFunction::LossFunction &);
      void updatePrediction(std::vector<float> & tempPredict, const std::vector<float> & newTreePrediction);

    };




    class GBSingleTreeGenerator{
    public:
      GBSingleTreeGenerator(const xgboost::data::SimpleSparseMatrix &traingData, std::vector<TreeNode> & tree):traingData_(traingData), newGBTree_(tree){};
      void trainANewTree(const std::vector<xgboost::detail::GradientPair> &gpairVec, const xgboost::parameters::ModelParam &param);
      const std::vector<float> getPrediction();

    private:
      std::vector<TreeNode> & newGBTree_;
      std::vector<int> position_{};
      std::vector<size_t> qexpand_{};
      size_t numOfNodes{};
      const xgboost::data::SimpleSparseMatrix & traingData_;
      void initData(const xgboost::parameters::ModelParam &param); //set initial values for the positions_ qxpand_ and push root into qexpand_
      void initNewNode(const std::vector<xgboost::detail::GradientPair> &gpairVec, const xgboost::parameters::ModelParam &param); // make leaf nodes for all qexpand, update node statistics, mark leaf value
      void findSplit();
      void resetPosition();
      void updateQueueExpand();
      void doPrune();
    };

  } //namespace tree
} //namespace xgboost
#endif //MINIXGBOOST_TREE_H
