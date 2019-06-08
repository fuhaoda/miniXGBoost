//
// Created by Haoda Fu on 2019-06-02.
//

#include "miniXGBoost.h"
#include "gbTreeEstimator.h"
#include "gbTreeEvaluator.h"

void miniXGBoost::MiniXGBoost::train(miniXGBoost::ModelParam &param,
                                     const miniXGBoost::data::DataSet &trainingData,
                                     const miniXGBoost::LossFunction &loss) {
  GBEstimator gbmModel(param, trainingData, loss);
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

  // loss from training data
  trainingLoss_ = gbmModel.trainingLoss();

  // predicted value on training data set (this is a copy assignment, might be expensive)
  yhatFromTrainingData_ = gbmModel.getPredictedValuesOnTrainingData();
}

void miniXGBoost::MiniXGBoost::evaluate(const miniXGBoost::data::DataSet &evaluationData,
                                        const Model &model, const miniXGBoost::LossFunction &loss) {
  GBEvaluator gbmEvaluator(evaluationData, model, loss);
  evaluationLoss_ = gbmEvaluator.getLoss();

  // predict value based on feature matrix (this is a copy assignment, might be expensive)
  yhat_ = gbmEvaluator.predict();
}
std::vector<float> miniXGBoost::MiniXGBoost::predict(const miniXGBoost::data::FeatureMatrix &featureMatrix,
                                                     const Model &model) {
  GBPredictor gbmPredictor(featureMatrix, model);

  // perfect forwarding. return predicted values
  return gbmPredictor.predict();

}
const miniXGBoost::Model &miniXGBoost::MiniXGBoost::getModel() const {
  return model_;
}

