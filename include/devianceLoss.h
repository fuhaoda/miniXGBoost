//
// Created by Haoda Fu on 2019-06-08.
//

#ifndef MINIXGBOOST_INCLUDE_DEVIANCELOSS_H_
#define MINIXGBOOST_INCLUDE_DEVIANCELOSS_H_

#include <cmath>
// Template for writing customized loss function
// Deviance loss is often used for binary classification
// The observed y is either 0 or 1, phat is the estimated probability
inline float squaredErrorLoss(float y, float phat) {
  return y>0.5f? -log(phat):-log(1-phat);
}

inline float squaredErrorGradient(float y, float phat) {
  return y>0.5f? -1.0f/phat: 1.0f/(1.0f-phat);
}

inline float squaredErrorHessian(float y, float phat) {
  return  y>0.5f? 1.0f/(phat*phat): 1.0f/((1.0f-phat)*(1.0f-phat));
}


#endif //MINIXGBOOST_INCLUDE_DEVIANCELOSS_H_
