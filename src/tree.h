#ifndef MINIXGBOOST_TREE_H
#define MINIXGBOOST_TREE_H

#include <limits> 

struct TreeNode {
  // Index of the parent, left and right child nodes. 
  int parent{-1}, lchild{-1}, rchild{-1}; 

  // Sum of the gradient/hessian of the samples associated with the treenode. 
  float sum_grad{0.0}, sum_hess{0.0};
  
  // Sum of the gradient/hessian associated with the future child during split
  // option enumeration. 
  float child_grad{0.0}, child_hess{0.0};
  
  // Last feature value examined. Used to set the split value.
  float last_value{0.0}; 
  
  // For an internal node of the tree, the index of the feature and the
  // associated value used to split the samples.
  int split_index{-1};
  float split_value{0.0}; 
    
  // Best score found when enumerating split options.
  float best_score{std::numeric_limits<float>::min()};  // 1.17549e-38
  
  // Default branch for missing value 
  bool missing_goto_right{true};
  
  // Reset stat
  void reset(); 
    
  // Update the best score.
  void update(size_t index, float grad, float hess, float fvalue, float eps,
              float thres, float lambda, float gamma, bool goto_right);

  void update(size_t index, float delta, float thres, float lambda,
              float gamma, bool goto_right); 

  // Return the weight of the tree node. 
  float weight(float lambda) { return -sum_grad / (sum_hess + lambda); } 
}; 
#endif

