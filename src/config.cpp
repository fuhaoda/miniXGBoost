#include "gbm.h"

ModelParam gbmParser(const std::string &config_file) {
  ModelParam param;

  // Open configuration file. 
  std::ifstream file(config_file);

  char comment = '#';
  char delimiter = '=';
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

    if (key.compare("num_round") == 0) {
      param.num_round = std::stoul(value);
    } else if (key.compare("gamma") == 0) {
      param.gamma = std::stof(value);
    } else if (key.compare("lambda") == 0) {
      param.lambda = std::stof(value);
    } else if (key.compare("max_depth") == 0) {
      param.max_depth = std::stoul(value);
    } else if (key.compare("eta") == 0) {
      param.eta = std::stof(value);
    } else if (key.compare("min_weight") == 0) {
      param.min_weight = std::stof(value);
    }
  }

  file.close();

  return param; 
}
