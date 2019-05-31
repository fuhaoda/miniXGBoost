#include "core.h"

std::vector<GBMCore> __model__; 

GBM::GBM(ModelParam &param, LossFunction &loss) {
  __model__.emplace_back(param, loss);
  id_ = __model__.size() - 1; 
}

void GBM::train(const std::string &train_data) const {
  __model__[id_].train(train_data); 
}

void GBM::apply(const std::string &test_data, std::vector<float> &pred) const {
  __model__[id_].apply(test_data, pred); 
}
