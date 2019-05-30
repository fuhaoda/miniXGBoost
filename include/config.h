#ifndef MINIXGBOOST_CONFIG_H
#define MINIXGBOOST_CONFIG_H

#include <string> 

struct ModelParam {
  // Reference: arXiv: 1603.02754v3

  // Number of gradient boosting trees to construct, variable K in Eq. (1) 
  size_t num_round = 10;

  // Parameters of the penalty term in Eq. (2). Additionally, by Eq. (7), gamma
  // is also used as the minimal loss change permitted by a split. 
  float gamma = 1.0;
  float lambda = 1.0; 

  // Maximum depth of each boosting tree. Depth 0 corresponds to the root layer.
  size_t max_depth = 3;

  // Shrinkage weights. Each newly added weights will be shrinked by eta after
  // each step of tree boosting (when a new layer of the tree is added). 
  float eta = 0.3; 

  // Minimum weight permitted in a leaf node.
  float min_weight = 1.0;
};

struct IOParam {
  // Path to load training data
  std::string training_data_path; 

  // Path to load testing data
  std::string testing_data_path; 
};

class ConfigParser {
public:  
  explicit ConfigParser(const std::string &config_file);

  // Return the model parameter
  ModelParam modelParam() const { return model_param_; } 

  // Return the io parameter
  IOParam ioParam() const { return io_param_; }
  
private:
  ModelParam model_param_;
  IOParam io_param_;
};

#endif 
