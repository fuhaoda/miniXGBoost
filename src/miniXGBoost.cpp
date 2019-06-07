//
// Created by Haoda Fu on 2019-06-02.
//

#include "miniXGBoost.h"
#include "gbTreeEstimator.h"
#include "gbTreeEvaluator.h"

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

  auto x = gbmModel.trainingLoss();
  int stop =1;
}


void miniXGBoost::MiniXGBoost::evaluate(const miniXGBoost::data::DataSet &evaluationData,
                                        const Model &model, const miniXGBoost::LossFunction &loss) {

GBEvaluator gbmEvaluator(evaluationData,model,loss);

auto lossValue = gbmEvaluator.getLoss();

int stop = 1;
}
std::vector<float> miniXGBoost::MiniXGBoost::predict(const miniXGBoost::data::FeatureMatrix &featureMatrix,
                                                     const miniXGBoost::LossFunction &loss) {
  return std::vector<float>();
}
const miniXGBoost::Model &miniXGBoost::MiniXGBoost::getModel() const {
  return model_;
}

