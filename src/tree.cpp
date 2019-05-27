//
// Created by Haoda Fu on 2019-05-25.
//

#include "tree.h"
#include "lossFunction.h"
#include <cmath>
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
    GBSingleTreeGenerator gbATree(traingData,GBRegModel.at(iter),gpairVec, param); //generate a new tree
    gbATree.trainANewTree();
//    updatePrediction(tempPred, gbATree.getPrediction()); //update the temp prediction vector
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

void xgboost::tree::GBSingleTreeGenerator::trainANewTree() {
  initData();
  initNewNode();
  for(size_t depth=0; depth < param_.max_depth; ++depth){
    findSplit();
    resetPosition();
    updateQueueExpand();
    initNewNode();
    if(qexpand_.empty()) break;
  }

  // set all the rest expanding nodes to leaf
  for( size_t i = 0; i < qexpand_.size(); ++ i ){
  }
  // start prunning the tree
  doPrune();
}

void xgboost::tree::GBSingleTreeGenerator::initData() {
  position_.clear();
  position_.resize(traingData_.sampleSize());
  std::fill(position_.begin(),position_.end(),0); //all samples point to the root node
  newGBTree_.clear();
  utils::myAssert(param_.max_depth<30,"Max depth of the tree exceed recommended depth! please change it <=20.");
  newGBTree_.reserve(1U<<param_.max_depth);
  qexpand_.clear();
  qexpand_.reserve(256);
  qexpand_.push_back(0); //root is qexpand_.
  numOfNodes =1; //now we have a root node.
}

void xgboost::tree::GBSingleTreeGenerator::initNewNode() {

  newGBTree_.resize(numOfNodes);

  for(size_t i=0; i< traingData_.sampleSize(); ++i){
    if(position_.at(i) < 0 ) continue;
    newGBTree_.at(position_.at(i)).sumGrad += gpairVec_.at(i).getGrad();
    newGBTree_.at(position_.at(i)).sumHess += gpairVec_.at(i).getHess();
  }

  for(size_t nid:qexpand_){
    newGBTree_.at(nid).calWeight(param_.reg_lambda);
    newGBTree_.at(nid).calGain(param_.reg_lambda);
  }

}


void xgboost::tree::TreeNode::calWeight(float lambda) {
  weight = -sumGrad/(sumHess+lambda);
}

void xgboost::tree::TreeNode::calGain(float lambda) {
  gain=0.5*sumGrad*sumGrad/(sumHess+lambda);
}

float xgboost::tree::TreeNode::getSomeGain(float sumG, float sumH, float lambda) {
  return 0.5*sumG*sumG/(sumH+lambda);
}

bool
xgboost::tree::TreeNode::updateBest(float loss_chg, unsigned split_index, float split_value, bool missing_GoToRight, float eps) {
  if(bestScore > loss_chg) return false;
  bestScore = loss_chg;
  splitIndex = split_index;
  splitValue= missing_GoToRight? split_value+eps:split_value-eps;
  missingGoToRight=missing_GoToRight;
  return true;
}


void xgboost::tree::GBSingleTreeGenerator::findSplit() {

  for(size_t feature=0; feature < traingData_.numOfCol(); ++feature){
    bool missingGoToRight=true;
    enumerateSplit(traingData_.getACol(feature), feature, missingGoToRight);
    missingGoToRight=false;
    enumerateSplit(traingData_.getAColRevese(feature), feature, missingGoToRight);
  }

  for(auto item:qexpand_){
    const auto nid = item;
    TreeNode & e = newGBTree_.at(nid);

    if(e.bestScore > param_.reg_lambda){
      addChilds(nid, e);
    } else {
      setLeaf(nid, e);
    }
  }
}

template <typename Iter>
void xgboost::tree::GBSingleTreeGenerator::enumerateSplit(Iter iter, size_t featureID, bool missingGoToRight) {
  // clean nodes split
  for(auto item:qexpand_){
    newGBTree_.at(item).sumSomeGrad=0;
    newGBTree_.at(item).sumSomeHess=0;
  }

  for(;iter!=iter.last(); ++iter){
    const auto rowID = iter.getItem().findex;
    const auto nid = position_.at(rowID);
    const auto fvalue = iter.getItem().fvalue;
    TreeNode & e = newGBTree_.at(nid);
    if(nid  < 0) continue;
    if(e.sumSomeHess==0){
      e.sumSomeGrad=gpairVec_.at(rowID).getGrad();
      e.sumSomeHess=gpairVec_.at(rowID).getHess();
      e.lastSplitValue = fvalue;
    } else {
      if(std::abs(fvalue-e.lastSplitValue) > param_.split_2eps && e.sumSomeHess >= param_.min_child_weight){
        const float csum_hess = e.sumHess-e.sumSomeHess;
        if(csum_hess >= param_.min_child_weight){
          const float csum_grad = e.sumGrad-e.sumSomeGrad;
          const float loss_chg = e.getSomeGain(e.sumSomeGrad, e.sumSomeHess, param_.reg_lambda)
                                 +e.getSomeGain(csum_grad, csum_hess, param_.reg_lambda)
                                 - e.gain;
          e.updateBest(loss_chg,featureID,(fvalue+e.lastSplitValue)*0.5f, missingGoToRight, param_.split_eps);
        }
      }
      e.sumSomeGrad+=gpairVec_.at(rowID).getGrad();
      e.sumSomeHess+=gpairVec_.at(rowID).getHess();
      e.lastSplitValue = fvalue;
    }
  }

}

void xgboost::tree::GBSingleTreeGenerator::addChilds(size_t nodeID,TreeNode &e) {
  numOfNodes = newGBTree_.size();
  newGBTree_.emplace_back();
  newGBTree_.emplace_back();
  e.leftChild= numOfNodes;
  e.rightChild =numOfNodes+1;
  newGBTree_.at(e.rightChild).parent=nodeID;
  newGBTree_.at(e.leftChild).parent=nodeID;
}

void xgboost::tree::GBSingleTreeGenerator::setLeaf(size_t nodeID, TreeNode &e) {
  e.weight=param_.learning_rate*e.weight;
  e.leftChild=-1;
  e.rightChild=-1;
  e.isLeaf=true;
}

void xgboost::tree::GBSingleTreeGenerator::resetPosition() {

  //reset position
  for(size_t i=0;i<traingData_.sampleSize();++i){
    const int nid=position_.at(i);
    TreeNode & e = newGBTree_.at(nid);
    if(nid < 0) continue;
    if(e.isLeaf){
      position_.at(i)=-1;
    } else {
      //set missing data to the correct position, correct others later
      position_.at(i)= e.missingGoToRight? e.rightChild:e.leftChild;
    }
  }

}
