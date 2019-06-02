//
// Created by Haoda Fu on 2019-05-18.
//

#include <iostream>
#include "miniXGBoost.h"
#include "squaredErrorLoss.h"

//
// Usage: ./demo config train.txt
//


int main(int argc, char *argv[]) {
  // Parse the configuration file to get model parameters.
  miniXGBoost::ModelParam param = miniXGBoost::configFileParser(argv[1]);

  // Specify the loss function
  miniXGBoost::LossFunction func{squaredErrorLoss, squaredErrorGradient, squaredErrorHessian};
  //miniXGBoost::MiniXGBoost();


  return 0;
}