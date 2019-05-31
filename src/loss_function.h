#ifndef MINIXGBOOST_LOSS_FUNCTION_H
#define MINIXGBOOST_LOSS_FUNCTION_H


using func_t = float (*)(float, float);

struct LossFunction {
  LossFunction(func_t f, func_t g, func_t h) :
    loss{f}, gradient{g}, hessian{h} { }

  func_t loss, gradient, hessian; 
};

inline float squaredErrorLoss(float y, float yhat) {
  return (y - yhat) * (y - yhat) / 0.5;
}

inline float squaredErrorGradient(float y, float yhat) {
  return yhat - y;
}

inline float squaredErrorGradient(float y, float yhat) {
  return 1.0;
} 

#endif 
