#include <fstream>
#include 
#include "config.h"

ConfigParser::ConfigParser(const std::string &config_file) {
  std::ifstream file(config_file);

  char delimiter = '=';
  char comment = '#';
  std::string line, key, value;

  while (!file.eof()) {
    // Read one line from the configuration file. 
    getline(file, line);

    // Anything beyond # is comment.
    line = line.substr(0, line.find(comment));

    // Find the delimiter.
    size_t dpos = line.find(delimiter);

    // Anything before the delimiter is the key.
    key = line.substr(0, dpos);

    // Anything after the delimiter is the value.
    value = line.substr(dpos + 1, line.length() - dpos - 1);

    // Skip a line starting with # or either key or value is empty.
    if (line.empty() || key.empty() || value.empty())
      continue;

    cleanString(key);
    cleanString(value);

    if (key.compare("num_round") == 0) {
      model_param_.num_round = std::stoul(value);
    } else if (key.compare("gamma") == 0) {
      model_param_.gamma = std::stof(value);
    } else if (key.compare("lambda") == 0) {
      model_param_.lambda = std::stof(value);
    } else if (key.compare("max_depth") == 0) {
      model_param_.max_depth = std::stoul(value);
    } else if (key.compare("eta") == 0) {
      model_param_.eta = std::stof(value);
    } else if (key.compare("min_weight") == 0) {
      model_param_.min_weight = std::stof(value);
    } else if (key.compare("training_data_path") == 0) {
      io_param_.training_data_path = value;
    } else if (key.compare("testing_data_path") == 0) {
      io_param_.testing_data_path = value;
    }
  }

  file.close();
}

