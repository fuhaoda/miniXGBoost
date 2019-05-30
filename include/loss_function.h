#ifndef MINIXGBOOST_LOSS_FUNCTION_H
#define MINIXGBOOST_LOSS_FUNCTION_H

// Abstract base class
class LossFunction {
public:
  // Parameter yhat is the predicted value. 
  virtual float gradient(float y, float yhat) = 0; 
  virtual float hessian(float y, float yhat) = 0;
  virtual ~LossFunction() { }
};

class SquaredErrorLoss : public LossFunction {
public:
  float gradient(float y, float yhat) override {
    return yhat - y;
  }

  float hessian(float y, float yhat) override {
    return 1.0;
  }

  ~SquaredErrorLoss() { } 
}; 

#endif 
