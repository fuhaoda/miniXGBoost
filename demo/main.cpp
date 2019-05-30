//
// Created by Haoda Fu on 2019-05-18.
//

#include <iostream>
#include <vector>
#include <data.h>
#include "tree.h"
#include "base.h"
#include "./config.h"
using namespace std;

namespace xgboost {
int CLIRunTask(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: <config>" << std::endl;
    return 0;
  }

  //load configuration parameters
  common::ConfigParse cp(argv[1]);
  auto cfg = cp.parse();
  cfg.emplace_back("seed", "0");

  for (int i = 2; i < argc; ++i) {
    char name[256], val[256];
    if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
      cfg.emplace_back(std::string(name), std::string(val));
    }
  }

  //load data
  data::SimpleSparseMatrix spMatrixTraining;
  spMatrixTraining.loadLibSVM("./machine.txt.train");
  spMatrixTraining.translateToCSCFormat();

  lossFunction::SquaredEorrLoss l2Loss;

  parameters::ModelParam mparam{};
  tree::GBTreeModel gbTreeModel(l2Loss);

  gbTreeModel.train(spMatrixTraining, mparam);


  return 0;
}

} //namespace xgboost





int main(int argc, char *argv[]) {

  xgboost::CLIRunTask(argc, argv);

  return 0;
}