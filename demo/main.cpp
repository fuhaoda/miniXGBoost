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

