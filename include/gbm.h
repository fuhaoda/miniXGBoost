#ifndef MINIXGBOOST_GBM_H
#define MINIXGBOOST_GBM_H

#include <string> 

// Public APIs of the Gradient Boosting Method. 

// Part #1: model parameters.
struct ModelParam {
  // Notations follow arXiv: 1603.02754v3

  // Number of gradient boosting trees to construct, Eq. (1)
  size_t K;

  // Parameters of the penalty term in Eq. (2). Additionally, by Eq. (7), gamma
  // is also used as the minimal loss change permitted by a split.
  float gamma = 1.0;
  float lambda = 1.0;

  // Maximum depth of each boosting tree. Depth 0 is the root layer.
  size_t max_depth = 3;
  
  // Shrinkage factor. Each newly added weights will be shrinked by eta after
  // each step of tree boosting, i.e., when a new layer of the tree is added.
  float eta = 0.3;

  // Minimum weight permitted in a leaf node.
  float min_weight = 1.0;
};

ModelParam gbmParser(const std::string &config_file); 

// Part #2: loss function 
using func_t = float (*)(float, float);

struct LossFunction {
  LossFunction(func_t f, func_t g, func_t h): loss{f}, grad{g}, hess{h} { }

  func_t loss, grad, hess;
};

// Part #3: Gradient Boosting Method
class GBM {
public:
  GBM(ModelParam &param, LossFunction &loss);

  // Train the model. 
  void train(const std::string &train_data) const; 

  // Apply the trained model to testing data set.
  void apply(const std::string &test_data, std::vector<float> &pred) const;

private:
  // Model ID
  size_t id_;  
}; 


#endif
