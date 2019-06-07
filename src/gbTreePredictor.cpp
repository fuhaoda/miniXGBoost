//
// Created by Haoda Fu on 2019-06-06.
//

#include "gbTreePredictor.h"
std::vector<float> miniXGBoost::GBPredictor::predict() {
  // initiate the prediction vector with mean average response
  std::vector<float> pre(fMatrix_.nRows(), model_.intercept);

  // update the prediction for each tree
  for (const auto &tree : model_.trees) {
    singleTreePrediction(tree, pre);
  }

  return pre;
}

void miniXGBoost::GBPredictor::singleTreePrediction(const std::vector<miniXGBoost::TreeNode> &tree,
                                                    std::vector<float> &prediction) {

  // for each observation (each row of feature matrix), find the leaf node, add the weight into
  // the prediction vector.
  for (size_t i = 0; i < prediction.size(); ++i) {
    const data::Entry *cbegin = fMatrix_.cbegin(i);
    const data::Entry *cend = fMatrix_.cend(i);
    size_t nid = 0;

    //  continue search until reach to the leaf, i.e. splitFeatureIndex == -1
    while (tree[nid].splitFeatureIndex >= 0) {
      for (auto iter = cbegin; iter != cend; ++iter) {
        if (iter->index == tree[nid].splitFeatureIndex) {
          nid = iter->value < tree[nid].splitValue ? tree[nid].lChild : tree[nid].rChild;
          break;
        } else if (iter == cend - 1) {
          // after searching all the feature id (iter == cend - 1), this row contains missing
          // feature. so we split it based on missing_go_to
          nid = tree[nid].missing_goto_right ? tree[nid].rChild : tree[nid].lChild;
          break;
        }
      }
    }
    prediction[i] += tree[nid].weight;
  }
}
