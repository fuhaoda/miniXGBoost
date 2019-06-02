//
// Created by Haoda Fu on 2019-06-01.
//

#ifndef MINIXGBOOST_INCLUDE_MINIXGBOOST_H_
#define MINIXGBOOST_INCLUDE_MINIXGBOOST_H_

#include <string>
#include <vector>

// Public APIs of the Gradient Boosting Method.

// Part #1: model parameters.
namespace miniXGBoost {
struct ModelParam {
  // Notations follow arXiv: 1603.02754v3

  // Number of gradient boosting trees to construct, parameter K Eq. (1)
  size_t nTrees{10};

  // Parameters of the penalty term in Eq. (2).
  // gamma is to penalize the number of tree nodes, lambda is to penalize the weights
  // Additionally, by Eq. (7), gamma
  // is also used as the minimal loss change permitted by a split.
  float reg_nodes = 1.0;
  float reg_weights = 1.0;

  // Maximum depth of each boosting tree. Depth 0 is the root layer.
  size_t max_depth = 3;

  // Shrinkage factor. Each newly added weights will be shrinked by eta after
  // each step of tree boosting, i.e., when a new layer of the tree is added.
  float shrinkage = 0.3;

  // Minimum sum of hessian to make a node eligible for split consideration.
  float min_weight = 1.0;
};

// Part #2: loss function
using func_t = float (*)(float, float);

struct LossFunction {
  LossFunction(func_t f, func_t g, func_t h) : loss{f}, grad{g}, hess{h} {}

  func_t loss, grad, hess;
};

// Part #3: MiniXGBoost Method TreeNode and Model
struct TreeNode {
  // Index of the parent, left and right child nodes.
  int parent{-1};
  int lChild{-1};
  int rChild{-1};

  //weight and gain
  float weight{0.0f};
  float gain{0.0f};

  size_t splitFeatureIndex{0};
  float splitValue{0};
};

class MiniXGBoost {
 public:
  MiniXGBoost(ModelParam &param, LossFunction &loss);

  // Train the model.
  void train(const std::string &train_data) const;

  // Evaluate the trained model to testing data set.
  void evaluate(const std::string &test_data, std::vector<float> &pred) const;

  // use the
  std::vector<float> predict(const std::string &featureMatrix);

 private:
  struct Model {
    float intercept{0};
    std::vector<std::vector<TreeNode>> trees{};
  } model_;
};

ModelParam configFileParser(const std::string &config_file);

} // namespace namespace miniXGBoost

#endif //MINIXGBOOST_INCLUDE_MINIXGBOOST_H_
