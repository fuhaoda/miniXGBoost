//
// Created by Haoda Fu on 2019-05-18.
//

#include <iostream>
#include <vector>
#include "base.h"
#include "./config.h"
using namespace std;



namespace xgboost{
  int CLIRunTask(int argc, char *argv[]){
    if(argc < 2){
      std::cout << "Usage: <config>" << std::endl;
      return 0;
    }

    common::ConfigParse cp(argv[1]);
    auto cfg=cp.parse();
    cfg.emplace_back("seed", "0");

    for(auto const &item : cfg){
      cout << item.first <<"=" << item.second << "!" <<endl;
    }



    for (int i = 2; i < argc; ++i) {
      char name[256], val[256];
      if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
        cfg.emplace_back(std::string(name), std::string(val));
      }
    }



    return 0;
  }

} //namespace xgboost





int main(int argc, char *argv[]){

  xgboost::CLIRunTask(argc, argv);




  return 0;
}