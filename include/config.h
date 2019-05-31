#ifndef MINIXGBOOST_CONFIG_H
#define MINIXGBOOST_CONFIG_H

#include <string> 
#include "core.h" 

// Parse the configuration 
class ConfigParser {
public:
  explicit ConfigParser(const std::string &config_file);

  // Return the model parameter
  ModelParam



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
