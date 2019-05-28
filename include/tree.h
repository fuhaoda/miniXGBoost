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

  bool isLeaf{false}; //this is important default setting

  // split values. Only valid for a parent node.
  float sumSomeGrad{0};
  float sumSomeHess{0};
  float getSomeGain(float sumG, float sumH, float lambda);
  bool updateBest(float loss_chg, unsigned split_index, float split_value, bool missing_GoToRight, float eps);
  size_t splitIndex{0};
  float splitValue{0};
  float lastSplitValue{std::numeric_limits<float>::min()};
  //true: build G_L based on ascending order of X_i, then G_R=G-G_L
  //false: build G_R based on descent order  of X_i, then G_L=G-G_R
  bool  missingGoToRight{true};
  float bestScore{0};
}; //end of TreeNode Struct definition


class GBTreeModel{
 public:
  void training(const data::SimpleSparseMatrix & traingData, const parameters::ModelParam & param);
  //  void testing(const data::SimpleSparseMatrix & testingData);
  //  std::vector<float> predicting(const data::SimpleSparseMatrix & featureData); //no response of y

 private:
  std::vector<std::vector<TreeNode>> GBRegModel;
  void updateGradHess(std::vector<detail::GradientPair> & gpair, const std::vector<float> & y, const std::vector<float> & tempPred, lossFunction::LossFunction &);
  void updatePrediction(std::vector<float> & tempPredict, const std::vector<float> & newTreePrediction);

};




class GBSingleTreeGenerator{
 public:
  GBSingleTreeGenerator(const xgboost::data::SimpleSparseMatrix &traingData, std::vector<TreeNode> & tree, const std::vector<xgboost::detail::GradientPair> &gpairVec, const xgboost::parameters::ModelParam &param):traingData_(traingData), newGBTree_(tree), param_{param}, gpairVec_{gpairVec}{};
  void trainANewTree();
  // const std::vector<float> getPrediction();

 private:
  std::vector<TreeNode> & newGBTree_;
  const parameters::ModelParam & param_;
  const std::vector<xgboost::detail::GradientPair> &gpairVec_;
  std::vector<int> position_{};
  std::vector<size_t> qexpand_{};
  size_t numOfNodes{};
  const xgboost::data::SimpleSparseMatrix & traingData_;
  void initData(); //set initial values for the positions_ qxpand_ and push root into qexpand_
  void initNewNode(); // make leaf nodes for all qexpand, update node statistics, mark leaf value
  void findSplit();
  template <typename Iter>
  void enumerateSplit(Iter iter, size_t featureID, bool missingGoToRight);
  void addChilds(size_t nodeID);
  void setLeaf(size_t nodeID);
  void resetPosition();
  void updateQueueExpand();
 //todo: add doPrune later, set a min_loss_change
  void doPrune(){};
};

} //namespace tree
} //namespace xgboost
#endif //MINIXGBOOST_TREE_H
