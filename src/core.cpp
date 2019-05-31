#include "core.h"
#include "io.h" 

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
  TreeNode &node = tree_[offset + iter]; 
  node.sum_grad = sum_grad;
  node.sum_hess = sum_hess;

  // Construct the tree
  for (size_t depth = 0; depth < param_.max_depth; ++depth) {
    // Find split index/value

    // Split the samples and update the prediction values for samples that
    // remain in the same treenodes.
  }  
}
