//
// Created by Haoda Fu on 2019-06-01.
//

#ifndef MINIXGBOOST_SRC_CORE_H_
#define MINIXGBOOST_SRC_CORE_H_

#include <vector>
#include "miniXGBoost.h"
#include "matrix.h"

namespace miniXGBoost {
class GBEstimator{
 public:
  GBEstimator(const ModelParam &param, const data::DataSet & data,  const
  LossFunction &func) : param_{param}, data_{data}, func_{func},
  max_nodes_{(1U << (param.max_depth + 1)) - 1}{model_.reserve(param.nTrees);}

  // Train the model.
  void train();

  //
  const std::vector<std::vector<FullTreeNode>> & getModel() const {return model_;}
  const float getIntercept() const {return intercept_;}
 private:
  // Parameter controlling floating point precision.
  const float eps{1e-5}, eps2{2e-5};


  // Model parameter. Copy the value passed in.
  ModelParam param_;

  // Loss function. Copy the value passed it.
  LossFunction func_;

  // Maximum number of tree nodes within each boosting tree.
  size_t max_nodes_;

  std::vector<std::vector<FullTreeNode>> model_{};
  float intercept_{};
  const data::DataSet & data_;

  // Internal working buffers, holding the current predictions of the
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
  // leaf nodes, the indices of which are in [first_node, last_node).
  // Split index -1 means the node will remain leaf and all associated
  // samples will be skipped for further consideration in the tree.
  void findSplit(size_t tid, size_t first_node, size_t last_node);

  // Helper function to findSplit.
  void enumSplit(size_t tid, size_t findex,
                 const data::Entry *cbegin, const data::Entry *cend, int incre,
                 size_t first_node, size_t last_node, bool goto_right,
                 float delta);

  // Split the tree nodes. Count the number of splits and the collect
  // the set of feature indices used for the split.
  size_t split(size_t tid, size_t first_node, size_t last_node, std::vector<size_t> &split_index);

  // Update sample locations. If the node a sample previously belongs
  // to is not split, update the prediction value of the sample. .
  void updatePos(size_t tid, const std::vector<size_t> &split_index);

  // Set the sum of gradient/hessian in the newly created leaf nodes.
  void setNewNodes(size_t tid);
};

}  // namespace miniXGBoost

#endif //MINIXGBOOST_SRC_CORE_H_
