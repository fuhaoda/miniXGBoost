#include "gbm.h"
#include "loss_function.h"

//
// Usage: ./demo config train.txt
//

int main(int argc, char **argv) {
  // Parse the configuration file to get model parameters.
  ModelParam param = gbmParser(argv[1]);

  // Specify the loss function
  LossFunction func{squaredErrorLoss, squaredErrorGradient, squaredErrorHessian};

  // Instantiate the model.
  GBM model(param, func);

  // Train the model.
  model.train(argv[2]); 

  return 0; 
}



// #include <iostream>
// #include <vector>
// #include <data.h>
// #include "tree.h"
// #include "base.h"
// #include "./config.h"
// using namespace std;



// namespace xgboost{
//   int CLIRunTask(int argc, char *argv[]){
//     if(argc < 2){
//       std::cout << "Usage: <config>" << std::endl;
//       return 0;
//     }

//     //load configuration parameters
//     common::ConfigParse cp(argv[1]);
//     auto cfg=cp.parse();
//     cfg.emplace_back("seed", "0");


//     for (int i = 2; i < argc; ++i) {
//       char name[256], val[256];
//       if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
//         cfg.emplace_back(std::string(name), std::string(val));
//       }
//     }

//     //load data
//     data::SimpleSparseMatrix spMatrix;
//     spMatrix.loadLibSVM("./machine.txt.train");
//     spMatrix.translateToCSCFormat();

//     parameters::ModelParam mparam{};
//     tree::GBTreeModel gbTreeModel;

//     gbTreeModel.training(spMatrix,mparam);

//     return 0;
//   }

// } //namespace xgboost





// int main(int argc, char *argv[]){

//   xgboost::CLIRunTask(argc, argv);




//   return 0;
// }
