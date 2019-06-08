//
// Created by Haoda Fu on 2019-05-18.
//

#include <iostream>
#include "miniXGBoost.h"
#include "squaredErrorLoss.h"
#include "io.h"
#include "../src/matrix.h"

//
// Usage: ./demo config
//


int main(int argc, char *argv[]) {
  // Parse the configuration file to get model parameters.
  miniXGBoost::ModelParam param = miniXGBoost::configFileParser(argv[1]);

  // Specify the loss function
  miniXGBoost::LossFunction func{squaredErrorLoss, squaredErrorGradient, squaredErrorHessian};

  // Load training data with column access support
  miniXGBoost::data::DataSet trainingData(true);
  miniXGBoost::dataIO::loadLibSVMData(trainingData, param.trainDataPath);

  miniXGBoost::MiniXGBoost myMiniXGBoost{};

  myMiniXGBoost.train(param, trainingData,func);
  auto model = myMiniXGBoost.getModel();



  miniXGBoost::data::DataSet evaluatingData(false);
  miniXGBoost::dataIO::loadLibSVMData(evaluatingData, param.evalDataPath);
  myMiniXGBoost.evaluate(evaluatingData,model, func);



  return 0;
}