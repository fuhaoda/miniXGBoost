//
// Created by Haoda Fu on 2019-06-01.
//

#include <numeric>
#include "gbTreeEstimator.h"

void miniXGBoost::GBEstimator::train() {
  intercept_ = std::accumulate(data_.y.begin(), data_.y.end(), 0.0f) / static_cast<float>(data_.y
      .size());

  model_.clear();
  model_.resize(param_.nTrees, std::vector<FullTreeNode>(max_nodes_));

  // Setup working buffers.
  pred_.clear();
  pred_.resize(data_.y.size(), intercept_);
  grad_.resize(data_.y.size());
  hess_.resize(data_.y.size());
  pos_.resize(data_.y.size());


  // Build gradient boosting trees.
  for (size_t k = 0; k < param_.nTrees; ++k) {
    // Compute the gradient and hessian
    float sum_grad = 0.0, sum_hess = 0.0;
    computeGradientHessian(sum_grad, sum_hess);

    // Put all samples to the root of the new tree to be constructed.
    std::fill(pos_.begin(), pos_.end(), 0);

    // Build a new tree and adjust prediction.
    createGBTree(k, sum_grad, sum_hess);
  }

}

void miniXGBoost::GBEstimator::computeGradientHessian(float &sum_grad, float &sum_hess) {
  for (size_t i = 0; i < pred_.size(); ++i) {
    grad_[i] = func_.grad(data_.y[i], pred_[i]);
    hess_[i] = func_.hess(data_.y[i], pred_[i]);
    sum_grad += grad_[i];
    sum_hess += hess_[i];
  }
}

void miniXGBoost::GBEstimator::createGBTree(size_t tid, float sum_grad, float sum_hess) {

  // Configure the root node of the new tree. Assume the tree is cleaned
  model_[tid][0].sum_grad = sum_grad;
  model_[tid][0].sum_hess = sum_hess;

  // Range of the indices of the tree nodes of each layer.
  size_t first_node = 0;
  size_t last_node = 1;

  // Construct the tree
  for (size_t depth = 0; depth < param_.max_depth; ++depth) {
    // Find split index/value
    findSplit(tid, first_node, last_node);

    // Perform splits.
    std::vector<size_t> split_index;
    auto nsplits = split(tid, first_node, last_node, split_index);

    // If no new split is required, terminate.
    if (nsplits == 0) break;

    // Update sample position.
    updatePos(tid, split_index);

    // Update the range of indices of the new leaf nodes.
    first_node = last_node;
    last_node += 2 * nsplits;

    // Set the sum of gradient/hessian of the new leaf nodes.
    setNewNodes(tid);
  }
  //todo save gain and weight into the nodes
  model_[tid].resize(last_node);
}

void miniXGBoost::GBEstimator::findSplit(size_t tid, size_t first_node, size_t last_node) {
  for (size_t col = 0; col < data_.X.nCols(); ++col) {
    // Get the [begin, end) of the column
    const data::Entry *cbegin = data_.X.cbegin(col, true);
    const data::Entry *cend = data_.X.cend(col, true);

    // Forward enumeration.
    enumSplit(tid, col, cbegin, cend, 1, first_node, last_node,
              true, eps);

    // Backward enumeration.
    enumSplit(tid, col, cend - 1, cbegin - 1, -1, first_node, last_node,
              false, -eps);
  }
}

void miniXGBoost::GBEstimator::enumSplit(size_t tid, size_t findex,
                                         const miniXGBoost::data::Entry *cbegin,
                                         const miniXGBoost::data::Entry *cend,
                                         int incre,
                                         size_t first_node,
                                         size_t last_node,
                                         bool goto_right,
                                         float delta) {
  // Reset statistics
  for (size_t iter = first_node; iter < last_node; ++iter)
    model_[tid][iter].reset();

  for (const data::Entry *entry = cbegin; entry != cend; entry += incre) {
    size_t row = entry->index;
    int nidx = pos_[row];

    if (nidx < 0)
      continue;

    float fvalue = entry->value;

    // Get a handle of the tree node.
    FullTreeNode &node = model_[tid][nidx];

    // Test if this is the first hit to update the statistics. The loss function
    // is convex, so the sum of hessian should be nonnegative.
    if (node.child_hess == 0.0) {
      node.child_grad = grad_[row];
      node.child_hess = hess_[row];
      node.last_value = fvalue;
    } else {
      node.update(findex, grad_[row], hess_[row], fvalue, eps2,
                  param_.min_weight, param_.reg_weights, param_.reg_nodes, goto_right);
    }
  }

  // The entire column is scanned. Do one final check because inside the for
  // loop each node does not know if the last value of the column has been
  // read. (due to this requirement fabsf(fvalue - last_value) > eps)
  for (size_t iter = first_node; iter < last_node; ++iter)
    model_[tid][iter].update(findex, delta, param_.min_weight, param_.reg_weights,
                             param_.reg_nodes, goto_right);

}

size_t miniXGBoost::GBEstimator::split(size_t tid,
                                       size_t first_node,
                                       size_t last_node,
                                       std::vector<size_t> &split_index) {
  // Within the current tree, [first_node, last_node) is the range of the
  // indices of the existing leaf nodes being examined. Any new leaf node will
  // be saved from index last_node onwards. The global index is obtained by
  // adding offset to the local index. The location to store the next new leaf
  // is tracked using variable curr.
  size_t nsplits = 0;
  size_t curr = last_node;

  for (size_t iter = first_node; iter < last_node; ++iter) {
    TreeNode &pnode = model_[tid][iter];
    if (pnode.splitFeatureIndex != -1) {
      ++nsplits;
      split_index.push_back(pnode.splitFeatureIndex);

      // Configure child nodes. Save left child first.
      pnode.lChild = curr;
      model_[tid][curr].parent = iter;
      ++curr;

      pnode.rChild = curr;
      model_[tid][curr].parent = iter;
      ++curr;
    }
  }

  std::sort(split_index.begin(), split_index.end());
  split_index.erase(std::unique(split_index.begin(), split_index.end()), split_index.end());
  return nsplits;
}

void miniXGBoost::GBEstimator::updatePos(size_t tid, const std::vector<size_t> &split_index) {
  for (size_t i = 0; i < pred_.size(); ++i) {
    int nidx = pos_[i];

    if (nidx < 0)
      continue;

    // Get a handle of the tree node.
    FullTreeNode &node = model_[tid][nidx];

    if (node.splitFeatureIndex == -1) {
      // The node remains leaf. Add the correction from the current boosting
      // tree to the prediction and mark pos to -1 to skip further
      // consideration.
      pred_[i] += node.weight(param_.reg_weights);
      pos_[i] = -1;
    } else {
      pos_[i] = node.missing_goto_right ? node.rChild : node.lChild;
    }
  }
    // Skip the else-branch. Ideally, we want to compare the ith sample's
    // feature value against the split value to assign its new
    // position. However, such random row access is inefficient for sparse
    // matrix storage. It is better to revisit each feature index that is
    // involved in the split and update.


  for (size_t col : split_index) {
    // Get the [begin, end) of the column
    const data::Entry *cbegin = data_.X.cbegin(col, true);
    const data::Entry *cend = data_.X.cend(col, true);

    for (const data::Entry *entry = cbegin; entry < cend; ++entry) {
      // Get current sample location.
      size_t row = entry->index;
      int nidx = pos_[row];

      if (nidx < 0)
        continue;

      TreeNode &node = model_[tid][nidx];
      float fvalue = entry->value;

      if (fvalue < node.splitValue) {
        pos_[row] = node.lChild;
      } else {
        pos_[row] = node.rChild;
      }
    }
  }
}

void miniXGBoost::GBEstimator::setNewNodes(size_t tid) {
  for (size_t i = 0; i < pred_.size(); ++i) {
    int nidx = pos_[i];

    if (nidx < 0)
      continue;

    model_[tid][nidx].sum_grad += grad_[i];
    model_[tid][nidx].sum_hess += hess_[i];
  }
}