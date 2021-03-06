//
// Created by Haoda Fu on 2019-06-01.
//

#ifndef MINIXGBOOST_INCLUDE_SQUAREDERRORLOSS_H_
#define MINIXGBOOST_INCLUDE_SQUAREDERRORLOSS_H_

// Template for writing customized loss function
// SquaredErrorLoss is often used for regression problem

inline float squaredErrorLoss(float y, float yhat) {
  return (y - yhat) * (y - yhat) / 2.0f;
}

inline float squaredErrorGradient(float y, float yhat) {
  return yhat - y;
}

inline float squaredErrorHessian(float y, float yhat) {
  return 1.0f;
}

#endif //MINIXGBOOST_INCLUDE_SQUAREDERRORLOSS_H_
