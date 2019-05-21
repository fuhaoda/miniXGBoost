//
// Created by Haoda Fu on 2019-05-18.
//

#include <iostream>
#include <vector>
#include <data.h>
#include <loadData.h>
#include "base.h"
#include "./config.h"
using namespace std;



namespace xgboost{
  int CLIRunTask(int argc, char *argv[]){
    if(argc < 2){
      std::cout << "Usage: <config>" << std::endl;
      return 0;
    }

    //load configuration parameters
    common::ConfigParse cp(argv[1]);
    auto cfg=cp.parse();
    cfg.emplace_back("seed", "0");


    for (int i = 2; i < argc; ++i) {
      char name[256], val[256];
      if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
        cfg.emplace_back(std::string(name), std::string(val));
      }
    }

    //load data
    data::SimpleSparseMatrix spMatrix;
    std::vector<float> y;
    data::LoadData dataLoader(spMatrix,y);
    dataLoader.loadLibSVM("/Users/haodafu/Documents/CodeDev/miniXGBoost/cmake-build-debug/demo/machine.txt.train");

    return 0;
  }

} //namespace xgboost





int main(int argc, char *argv[]){

  xgboost::CLIRunTask(argc, argv);




  return 0;
}