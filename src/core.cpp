#include <algorithm> 
#include "core.h"
#include "io.h" 

const float GBMCore::eps = 1e-5;
const float GBMCore::eps2 = 2e-5; 

void GBMCore::train(const std::string &train_data) {
  // Load training data set.
  std::vector<std::vector<Entry>> feature_matrix; 
  loadSVMData(train_data, feature_matrix, resp_);

  // Save the feature matrix in both CSR and CSC format. For CSC format, each
  // column is further sorted by value.  
  matrix_ = SparseMatrix(feature_matrix, true, true);

  // Reserve memory for trees.
  tree_.resize(param_.K * max_nodes_); 

  // Setup working buffers. 
  pred_.resize(resp_.size());
  grad_.resize(resp_.size());
  hess_.resize(resp_.size());
  pos_.resize(resp_.size()); 

  // Set the initial prediction to zero. 
  std::fill(pred_.begin(), pred_.end(), 0.0);
  
  // Build gradient boosting trees.
  for (size_t k = 0; k < param_.K; ++k) {
    // Compute the gradient and hessian
    float sum_grad = 0.0, sum_hess = 0.0;      
    computeGradientHessian(sum_grad, sum_hess);

    // Put all samples to the root of the new tree to be constructed. 
    std::fill(pos_.begin(), pos_.end(), 0); 
    
    // Build a new tree and adjust prediction.
    createGBTree(k, sum_grad, sum_hess); 
  } 
}

void GBMCore::apply(const std::string &test_data,
                    std::vector<float> &pred) const {

}

void GBMCore::computeGradientHessian(float &sum_grad, float &sum_hess) {
  for (size_t i = 0; i < pred_.size(); ++i) {
    grad_[i] = func_.grad(resp_[i], pred_[i]);
    hess_[i] = func_.hess(resp_[i], pred_[i]);

    sum_grad += grad_[i];
    sum_hess += hess_[i]; 
  }
} 

void GBMCore::createGBTree(size_t tid, float sum_grad, float sum_hess) {
  // Offset to save tree nodes of the new tree.
  size_t offset = tid * max_nodes_;

  // Number of tree nodes created on the new tre.
  int iter = 0;

  // Configure the root node of the new tree.
  tree_[offset].sum_grad = sum_grad;
  tree_[offset].sum_hess = sum_hess; 

  // Range of the indices of the tree nodes of each layer.
  size_t first_node = 0;
  size_t last_node = 1; 

  // Construct the tree
  for (size_t depth = 0; depth < param_.max_depth; ++depth) {
    // Find split index/value
    findSplit(offset, first_node, last_node); 

    // Perform splits.
    int nsplits = 0;
    std::vector<size_t> split_index; 
    split(offset, first_node, last_node, nsplits, split_index);

    // If no new split is required, terminate. 
    if (!nsplits)
      break; 

    // Update sample position.
    updatePos(offset, split_index); 

    // Update the range of indices of the new leaf nodes.
    first_node = last_node;
    last_node += 2 * nsplits;

    // Set the sum of gradient/hessian of the new leaf nodes.
    setNewNodes(offset); 
  }  
}

void GBMCore::findSplit(size_t offset, size_t first_node, size_t last_node) {
  for (size_t col = 0; col < matrix_.nCols(); ++col) {
    // Get the [begin, end) of the column
    const Entry *cbegin = matrix_.begin(col, true);
    const Entry *cend = matrix_.begin(col, true);

    // Forward enumeration.
    enumSplit(col, offset, cbegin, cend - 1, 1, first_node, last_node,
              true, eps);

    // Backward enumeration.
    enumSplit(col, offset, cend - 1, cbegin, -1, first_node, last_node,
              false, -eps); 
  }
}

void GBMCore::enumSplit(size_t findex, size_t offset,
                        const Entry *cbegin, const Entry *cend, int incre,
                        size_t first_node, size_t last_node, bool goto_right,
                        float delta) {
  // Reset statistics
  for (size_t iter = offset + first_node; iter < offset + last_node; ++iter)
    tree_[iter].reset();

  for (const Entry *entry = cbegin; entry <= cend; entry += incre) {
    size_t row = entry->index;
    int nidx = pos_[row];

    if (nidx < 0)
      continue;

    float fvalue = entry->value;

    // Get a handle of the tree node.
    TreeNode &node = tree_[offset + nidx];

    // Test if this is the first hit to update the statistics. The loss function
    // is convex, so the sum of hessian should be nonnegative.
    if (node.child_hess == 0.0) {
      node.child_grad = grad_[row];
      node.child_hess = hess_[row];
      node.last_value = fvalue;
    } else {
      node.update(findex, grad_[row], hess_[row], fvalue, eps2,
                  param_.min_weight, param_.lambda, param_.gamma, goto_right);
    }
  }

  // The entire column is scanned. Do one final check because inside the for
  // loop each node does not know if the last value of the column has been
  // read.
  for (size_t iter = offset + first_node; iter < offset + last_node; ++iter)
    tree_[iter].update(findex, delta, param_.min_weight, param_.lambda,
                       param_.gamma, goto_right);
}

void GBMCore::split(size_t offset, size_t first_node, size_t last_node,
                    int &nsplits, std::vector<size_t> &split_index) {
  // Within the current tree, [first_node, last_node) is the range of the
  // indices of the existing leaf nodes being examined. Any new leaf node will
  // be saved from index last_node onwards. The global index is obtained by
  // adding offset to the local index. The location to store the next new leaf
  // is tracked using variable curr. 
  size_t curr = last_node; 

  for (size_t iter = first_node; iter < last_node; ++iter) {
    TreeNode &pnode = tree_[offset + iter];
    if (pnode.split_index != -1) {
      nsplits++;
      split_index.push_back(pnode.split_index); 

      // Configure child nodes. Save left child first.
      pnode.lchild = curr;
      tree_[offset + curr].parent = iter;
      curr++;

      pnode.rchild = curr;
      tree_[offset + curr].parent = iter;
      curr++;
    }
  }

  std::sort(split_index.begin(), split_index.end());
  split_index.resize(std::unique(split_index.begin(), split_index.end()) -
                     split_index.begin()); 
} 

void GBMCore::updatePos(size_t offset, const std::vector<size_t> &split_index) {
  for (size_t i = 0; i < pred_.size(); ++i) {
    int nidx = pos_[i];

    if (nidx < 0)
      continue;

    // Get a handle of the tree node.
    TreeNode &node = tree_[offset + nidx];

    if (node.split_index == -1) {
      // The node remains leaf. Add the correction from the current boosting
      // tree to the prediction and mark pos to -1 to skip further
      // consideration.
      pred_[i] += node.weight(param_.lambda);
      pos_[i] = -1;
    }

    // Skip the else-branch. Ideally, we want to compare the ith sample's
    // feature value against the split value to assign its new
    // position. However, such random row access is inefficient for sparse
    // matrix storage. It is better to revisit each feature index that is
    // involved in the split and update.
  }

  for (size_t i = 0; i < split_index.size(); ++i) {
    size_t col = split_index[i];

    // Get the [begin, end) of the column
    const Entry *cbegin = matrix_.begin(col, true);
    const Entry *cend = matrix_.begin(col, true);

    for (const Entry *entry = cbegin; entry < cend; ++entry) {
      // Get current sample location.
      size_t row = entry->index; 
      int nidx = pos_[row]; 

      if (nidx < 0)
        continue;
      
      TreeNode &node = tree_[offset + nidx];
      float fvalue = entry->value;

      if (fvalue < node.split_value) {
        pos_[row] = node.lchild;        
      } else {
        pos_[row] = node.rchild;
      }
    }
  }
} 

void GBMCore::setNewNodes(size_t offset) {
  for (size_t i = 0; i < pred_.size(); ++i) {
    int nidx = pos_[i];

    if (nidx < 0)
      continue;

    tree_[offset + nidx].sum_grad += grad_[i];
    tree_[offset + nidx].sum_hess += hess_[i]; 
  } 
} 
