//
// Created by Haoda Fu on 2019-06-02.
//

#include "miniXGBoost.h"
#include "gbTreeEstimator.h"

void miniXGBoost::MiniXGBoost::train(miniXGBoost::ModelParam &param,
                                     const miniXGBoost::data::DataSet &trainingData,
                                     const miniXGBoost::LossFunction &loss) {
  GBEstimator gbmModel(param, trainingData,loss);
  gbmModel.train();
  auto model = gbmModel.getModel();

  //save the necessary tree information into MiniGBoost instance
  model_.trees.clear();
  model_.trees.resize(param.nTrees);

  for (size_t i = 0; i < param.nTrees; ++i) {
    //only copy the base class information from the derived class
    model_.trees[i].assign(model[i].begin(), model[i].end());
  }

  // save the intercept
  model_.intercept = gbmModel.getIntercept();
}


void miniXGBoost::MiniXGBoost::evaluate(const miniXGBoost::data::DataSet &evaluationData,
                                        const miniXGBoost::LossFunction &loss) {

}
std::vector<float> miniXGBoost::MiniXGBoost::predict(const miniXGBoost::data::FeatureMatrix &featureMatrix,
                                                     const miniXGBoost::LossFunction &loss) {
  return std::vector<float>();
}

