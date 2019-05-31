#ifndef MINIXGBOOST_LOSS_FUNCTION_H
#define MINIXGBOOST_LOSS_FUNCTION_H

// Template for loss function

inline float squaredErrorLoss(float y, float yhat) {
  return (y - yhat) * (y - yhat) / 0.5;
}

inline float squaredErrorGradient(float y, float yhat) {
  return yhat - y;
}

inline float squaredErrorHessian(float y, float yhat) {
  return 1.0;
}

#endif 
