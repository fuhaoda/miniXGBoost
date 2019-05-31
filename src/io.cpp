#include <cstdio>
#include <sstream>
#include <fstream>
#include "io.h"

void loadSVMData(const std::string &fname,
                 std::vector<std::vector<Entry>> &feature_matrix,
                 std::vector<float> &response) {
  std::ifstream file(fname); 
  std::istringstream ss;
  std::string line, field;

  while (getline(file, line)) {
    ss.clear();
    ss.str(line);      

    // Parse the line
    std::vector<Entry> entries;
    size_t index;
    float value; 
    while(getline(ss, field, ' ')) {
      // Skip extra white space
      if (field.empty())
        continue; 
      
      if (sscanf(field.c_str(), "%zu:%f", &index, &value) == 2) {
        entries.emplace_back(index, value);
      } else if (sscanf(field.c_str(), "%f", &value) == 1) {
        response.push_back(value);
      }
    }

    feature_matrix.push_back(entries);
  }

  file.close();
}
