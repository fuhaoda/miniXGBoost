//
// Created by Haoda Fu on 2019-06-02.
//

#include "treeNodes.h"
#include <cmath>

void miniXGBoost::FullTreeNode::reset() {
  child_grad = child_hess = last_value = 0.0;
}

void miniXGBoost::FullTreeNode::update(size_t index,
                                       float grad,
                                       float hess,
                                       float fvalue,
                                       float eps,
                                       float thres,
                                       float lambda,
                                       float gamma,
                                       bool goto_right) {
  // Compute the sum of gradient/hessian that would go to the other child node.
  double other_grad = sum_grad - child_grad;
  double other_hess = sum_hess - child_hess;

  // Consider the change in the loss function if the difference in the feature
  // value is distinguishable and the hessian accumulated in the child node has
  // exceeded the minimum threshold.
  if (fabsf(fvalue - last_value) > eps && child_hess >= thres &&
      other_hess >= thres) {
    // Compute the loss change
    double loss_change = pow(child_grad, 2) / (child_hess + lambda) +
        pow(other_grad, 2) / (other_hess + lambda) -
        pow(sum_grad, 2) / (sum_hess + lambda);

    loss_change = loss_change / 2.0f - gamma;

    if (loss_change > best_score) {
      splitFeatureIndex = index;
      splitValue = 0.5f * (last_value + fvalue);
      best_score = loss_change;
      missing_goto_right = goto_right;
    }
  }

  child_grad += grad;
  child_hess += hess;
  last_value = fvalue;
}

void miniXGBoost::FullTreeNode::update(size_t index,
                                       float delta,
                                       float thres,
                                       float lambda,
                                       float gamma,
                                       bool goto_right) {
  // Compute the sum of gradient/hessian that would go to the other child node.
  double other_grad = sum_grad - child_grad;
  double other_hess = sum_hess - child_hess;

  if (child_hess >= thres && other_hess >= thres) {
    // Compute the loss change
    double loss_change = pow(child_grad, 2) / (child_hess + lambda) +
        pow(other_grad, 2) / (other_hess + lambda) -
        pow(sum_grad, 2) / (sum_hess + lambda);

    loss_change = loss_change / 2.0f - gamma;

    if (loss_change > best_score) {
      splitFeatureIndex  = index;
      splitValue = last_value + delta;
      missing_goto_right = goto_right;
    }
  }
}
