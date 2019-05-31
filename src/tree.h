#ifndef MINIXGBOOST_TREE_H
#define MINIXGBOOST_TREE_H

#include <limits> 

struct TreeNode {
  // Index of the parent, left and right child nodes. 
  int parent{-1}, lchild{-1}, rchild{-1}; 

  // If the treenode is a leaf.
  bool is_leaf{true}; 

  // Sum of the gradient/hessian of the samples associated with the treenode. 
  float sum_grad{0.0}, sum_hess{0.0};

  // For an internal node of the tree, the index of the feature and the
  // associated value used to split the samples.
  int split_index{-1};
  float split_value{0.0}; 

  // Best score found when enumerating split options.
  float best_score{std::numeric_limits<float>::min()}; 
  
  // Default branch for missing value 
  bool default_to_right{true};

  // Update the best score
  void updateBest(float score, size_t sindex, float svalue, bool go_right) { } 
  
  // Return the weight of the tree node. 
  float weight(float lambda) { return -sum_grad / (sum_hess + lambda); } 

  // Return the gain, this is the individual term inside the bracket of Eq. (7)
  // in arXiv: 1603.02754v
  float gain(float lambda) { return sum_grad * sum_grad / (sum_hess + lambda); }  
}; 



// #include <iostream>
// #include <vector>
// #include "data.h"
// #include "base.h"
// #include "lossFunction.h"

// namespace xgboost{
// namespace tree{

// struct TreeNode{
//   // split values. Only valid for a parent node.
//   float sumSomeGrad{0};
//   float sumSomeHess{0};
//   float getSomeGain(float sumG, float sumH, float lambda);
//   bool updateBest(float loss_chg, unsigned split_index, float split_value, bool missing_GoToRight, float eps);
//   float lastSplitValue{std::numeric_limits<float>::min()};
//   //true: build G_L based on ascending order of X_i, then G_R=G-G_L
//   //false: build G_R based on descent order  of X_i, then G_L=G-G_R
//   float bestScore{0};
// }; //end of TreeNode Struct definition


// class GBTreeModel{
//  public:
//   void training(const data::SimpleSparseMatrix & traingData, const parameters::ModelParam & param);
//   //  void testing(const data::SimpleSparseMatrix & testingData);
//   //  std::vector<float> predicting(const data::SimpleSparseMatrix & featureData); //no response of y

//  private:
//   std::vector<std::vector<TreeNode>> GBRegModel;
//   void updateGradHess(std::vector<detail::GradientPair> & gpair, const std::vector<float> & y, const std::vector<float> & tempPred, lossFunction::LossFunction &);
//   void updatePrediction(std::vector<float> & tempPredict, const std::vector<float> & newTreePrediction);

// };




// class GBSingleTreeGenerator{
//  public:
//   GBSingleTreeGenerator(const xgboost::data::SimpleSparseMatrix &traingData, std::vector<TreeNode> & tree, const std::vector<xgboost::detail::GradientPair> &gpairVec, const xgboost::parameters::ModelParam &param):traingData_(traingData), newGBTree_(tree), param_{param}, gpairVec_{gpairVec}{};
//   void trainANewTree();
//   // const std::vector<float> getPrediction();

//  private:
//   std::vector<TreeNode> & newGBTree_;
//   const parameters::ModelParam & param_;
//   const std::vector<xgboost::detail::GradientPair> &gpairVec_;
//   std::vector<int> position_{};
//   std::vector<size_t> qexpand_{};
//   size_t numOfNodes{};
//   const xgboost::data::SimpleSparseMatrix & traingData_;
//   void initData(); //set initial values for the positions_ qxpand_ and push root into qexpand_
//   void initNewNode(); // make leaf nodes for all qexpand, update node statistics, mark leaf value
//   void findSplit();
//   template <typename Iter>
//   void enumerateSplit(Iter iter, size_t featureID, bool missingGoToRight);
//   void addChilds(size_t nodeID,TreeNode &e);
//   void setLeaf(size_t nodeID, TreeNode &e);
//   void resetPosition();
//   void updateQueueExpand();
//  //todo: add doPrune later, set a min_loss_change
//   void doPrune(){};
// };

// } //namespace tree
// } //namespace xgboost
#endif //MINIXGBOOST_TREE_H
