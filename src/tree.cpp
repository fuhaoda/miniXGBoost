//
// Created by Haoda Fu on 2019-05-25.
//

#include "tree.h"
#include "lossFunction.h"

void xgboost::tree::GBTreeModel::training(const xgboost::data::SimpleSparseMatrix &traingData,
                                          const xgboost::parameters::ModelParam &param) {

  std::vector<float> tempPred(traingData.sampleSize(), traingData.overallResponseMean());
  std::vector<detail::GradientPair> gpairVec(traingData.sampleSize());
  auto & y = traingData.getY();

  lossFunction::SquaredEorrLoss l2Loss;

  GBRegModel.clear();
  GBRegModel.resize(param.num_round);

  for(int iter=0; iter < param.num_round; ++iter){
    updateGradHess(gpairVec, y, tempPred, l2Loss); //calculate the gradient
    GBSingleTreeGenerator gbATree(traingData,GBRegModel.at(iter)); //generate a new tree
    gbATree.trainANewTree(gpairVec, param);
    updatePrediction(tempPred, gbATree.getPrediction()); //update the temp prediction vector
  }
}


void xgboost::tree::GBTreeModel::updateGradHess(std::vector<xgboost::detail::GradientPair> &gpairVec,
                                                const std::vector<float> & y, const std::vector<float> &tempPred,
                                                xgboost::lossFunction::LossFunction & lossFun) {
  utils::myAssert(gpairVec.size()==tempPred.size(),"The size of the gradient function does not match the size of response y!");
  for(size_t i=0; i<tempPred.size();++i){
    gpairVec.at(i).setGrad(lossFun.gradient(y.at(i),tempPred.at(i)));
    gpairVec.at(i).setHess(lossFun.hessian(y.at(i),tempPred.at(i)));
  }
}

void xgboost::tree::GBTreeModel::updatePrediction(std::vector<float> &tempPredict,
                                                  const std::vector<float> &newTreePrediction) {
  for(size_t iter=0; iter < tempPredict.size(); ++iter){
    tempPredict.at(iter)+=newTreePrediction.at(iter);
  }
}

void xgboost::tree::GBSingleTreeGenerator::trainANewTree(const std::vector<xgboost::detail::GradientPair> &gpairVec,
                                                         const xgboost::parameters::ModelParam &param) {
  initData(param);
  initNewNode(gpairVec, param);
  for(size_t depth=0; depth < param.max_depth; ++depth){
    findSplit();
    resetPosition();
    updateQueueExpand();
    initNewNode(gpairVec,param);
    if(qexpand_.empty()) break;
  }

  // set all the rest expanding nodes to leaf
  for( size_t i = 0; i < qexpand_.size(); ++ i ){
  }
  // start prunning the tree
  doPrune();
}

void xgboost::tree::GBSingleTreeGenerator::initData(const xgboost::parameters::ModelParam &param) {
  position_.clear();
  position_.resize(traingData_.sampleSize());
  std::fill(position_.begin(),position_.end(),0); //all samples point to the root node
  newGBTree_.clear();
  utils::myAssert(param.max_depth<30,"Max depth of the tree exceed recommended depth! please change it <=20.");
  newGBTree_.reserve(1U<<param.max_depth);
  qexpand_.clear();
  qexpand_.reserve(256);
  qexpand_.push_back(0); //root is qexpand_.
  numOfNodes =1; //now we have a root node.
}

void xgboost::tree::GBSingleTreeGenerator::initNewNode(const std::vector<xgboost::detail::GradientPair> &gpairVec, const xgboost::parameters::ModelParam &param) {

  newGBTree_.resize(numOfNodes);

  for(size_t i=0; i< traingData_.sampleSize(); ++i){
    if(position_.at(i) < 0 ) continue;
    newGBTree_.at(position_.at(i)).sumGrad += gpairVec.at(i).getGrad();
    newGBTree_.at(position_.at(i)).sumHess += gpairVec.at(i).getHess();
  }

  for(size_t nid:qexpand_){
    newGBTree_.at(nid).calWeight(param.reg_lambda);
    newGBTree_.at(nid).calGain(param.reg_lambda);
  }

}


void xgboost::tree::TreeNode::calWeight(float lambda) {
weight = -sumGrad/(sumHess+lambda);
}

void xgboost::tree::TreeNode::calGain(float lambda) {
gain=0.5*sumGrad*sumGrad/(sumHess+lambda);
}
