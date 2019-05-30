//
// Created by Haoda Fu on 2019-05-25.
//

#ifndef MINIXGBOOST_LOSSFUNCTION_H
#define MINIXGBOOST_LOSSFUNCTION_H
namespace xgboost::lossFunction {
//for simplicity, the functions are passed by value
class LossFunction {
 public:
  virtual float loss(float y, float yhat) = 0;
  virtual float gradient(float y, float yhat) = 0;
  virtual float hessian(float y, float yhat) = 0;
  virtual std::string getLossFunctionName(){ return {}; };
  virtual ~LossFunction() = default;
};

class SquaredEorrLoss : public LossFunction {
 public:
  float loss(float y, float yhat) override {
    return (y-yhat)*(y-yhat);
  }
  float gradient(float y, float yhat) override {
    return yhat - y;
  }

  float hessian(float y, float yhat) override {
    return 1.0;
  }

  std::string getLossFunctionName() override {
    return "Squared Error Loss";
  }
  ~SquaredEorrLoss() = default;
};
}

#endif //MINIXGBOOST_LOSSFUNCTION_H
