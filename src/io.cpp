#include <cstdio>
#include <sstream>
#include <fstream>
#include <string>
#include <limits>
#include "io.h"


using myfloat_t = float;




// TODO: labels -> responses

void loadSVMData(const std::string &fname,
                 std::vector<std::vector<Entry>> &feature_matrix,
                 std::vector<float> &labels) {
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
      if (sscanf(field.c_str(), "%zu:%f", &index, &value) == 2) {
        entries.emplace_back(index, value);
      } else if (sscanf(field.c_str(), "%f", &value) == 1) {
        labels.push_back(value);
      }
    }

    feature_matrix.push_back(entries);
  }

  file.close();
}
