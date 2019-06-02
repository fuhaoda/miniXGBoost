//
// Created by Haoda Fu on 2019-05-18.
//

#include <iostream>
#include "miniXGBoost.h"
#include "squaredErrorLoss.h"
#include "io.h"
#include "../src/matrix.h"

//
// Usage: ./demo config train.txt
//


int main(int argc, char *argv[]) {
  // Parse the configuration file to get model parameters.
  miniXGBoost::ModelParam param = miniXGBoost::configFileParser(argv[1]);

  // Specify the loss function
  miniXGBoost::LossFunction func{squaredErrorLoss, squaredErrorGradient, squaredErrorHessian};

  // Load training data with column access support
  miniXGBoost::data::DataSet trainingData(true);
  miniXGBoost::dataIO::loadLibSVMData(trainingData, param.trainDataPath);

  // Training

  //miniXGBoost::MiniXGBoost();


  // Evaluating
  miniXGBoost::data::DataSet evaluatingData(false);
  // Prediction
  miniXGBoost::data::FeatureMatrix featureMatrix(false);

  return 0;
}