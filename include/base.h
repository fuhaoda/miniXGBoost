//
// Created by Haoda Fu on 2019-05-18.
//

#ifndef MINXGBOOST_BASE_H
#define MINXGBOOST_BASE_H

#include <iostream>

namespace xgboost::utils {

inline void error(const std::string &msg) {
  fprintf(stderr, "Error: %s\n", msg.c_str());
  exit(-1);
}

inline void myAssert(bool exp) {
  if (!exp) error("Assert Error!");
}

inline void myAssert(bool exp, const std::string &msg) {
  if (!exp) error(msg);
}

inline void warning(const std::string &msg) {
  fprintf(stderr, "Warning: %s \n", msg.c_str());
}

inline void warning(bool exp, const std::string &msg) {
  if (exp) fprintf(stderr, "Warning: %s \n", msg.c_str());
}

template<typename T>
class ForwardIterator {
 public:
  ForwardIterator(const T *begin, const T *end) : begin_{begin}, dprt_{begin}, end_{end} {}
  bool operator==(const T *rhs) const {
    return dprt_ == rhs;
  }
  bool operator!=(const T *rhs) const {
    return dprt_ != rhs;
  }
  const T *operator++() {
    return ++dprt_;
  }
  const T &getItem() const {
    return *dprt_;
  }
  const T *begin() const {
    return begin_;
  }
  const T *end() const {
    return end_;
  }
  // point to last item
  const T *last() const {
    return end_;
  }

 private:
  const T *begin_, *dprt_, *end_;
};

template<typename T>
class BackwardIterator {
 public:
  BackwardIterator(const T *begin, const T *end) : rbegin_{end - 1},
                                                   dprt_{end - 1},
                                                   rend_{begin - 1} {}
  bool operator==(const T *rhs) const {
    return dprt_ == rhs;
  }
  bool operator!=(const T *rhs) const {
    return dprt_ != rhs;
  }
  const T *operator++() {
    return --dprt_;
  }
  const T &getItem() const {
    return *dprt_;
  }
  const T *rbegin() const {
    return rbegin_;
  }
  const T *rend() const {
    return rend_;
  }
  // point to last item
  const T *last() const {
    return rend_;
  }
 private:
  const T *rbegin_, *dprt_, *rend_;
};

} //namespace utils


namespace xgboost::parameters {
struct ModelParam {
  size_t num_round{10};
  float reg_lambda{1};
  size_t max_depth{3}; //including the root layer
  float learning_rate{0.3};
  size_t minSamplePerNode{3};
  //less important parameters below
  float min_child_weight{1.0};
  const float split_eps = 1e-5f;
  const float split_2eps = 2 * split_eps;
};
struct IOParam {
  std::string pathTraining{};
  std::string pathTesting{};
};
}

namespace xgboost::detail {
/*! \brief Implementation of gradient statistics pair. Template specialisation
* may be used to overload different gradients types e.g. low precision, high
* precision, integer, floating point. */
template<typename T>
class GradientPairInternal {

 public:
  //constructor
  GradientPairInternal() : grad_(0), hess_(0) {}
  GradientPairInternal(float grad, float hess) : grad_(grad), hess_(hess) {}
  //copy constructor
  GradientPairInternal(const GradientPairInternal<T> &gpair)
      : grad_(gpair.grad_), hess_(gpair.hess_) {}

  void setGrad(float grad) { grad_ = grad; }
  void setHess(float hess) { hess_ = hess; }
  float getGrad() const { return grad_; }
  float getHess() const { return hess_; }

  //define operator += and -=. This operation is preferred with the original copy is not needed.
  GradientPairInternal<T> &operator+=(const GradientPairInternal<T> &rhs) {
    grad_ += rhs.grad_;
    hess_ += rhs.hess_;
    return *this;
  }

  GradientPairInternal<T> &operator-=(const GradientPairInternal<T> &rhs) {
    grad_ -= rhs.grad_;
    hess_ -= rhs.hess_;
    return *this;
  }

  //define operator + and -. This is used when we need to assign the results to a new gradient pair.
  GradientPairInternal<T> &operator+(const GradientPairInternal<T> &rhs) {
    GradientPairInternal<T> gpair;
    gpair.grad_ = grad_ + rhs.grad_;
    gpair.hess_ = hess_ + rhs.hess_;
    return gpair;
  }

  GradientPairInternal<T> &operator-(const GradientPairInternal<T> &rhs) {
    GradientPairInternal<T> gpair;
    gpair.grad_ = grad_ - rhs.grad_;
    gpair.hess_ = hess_ - rhs.hess_;
    return gpair;
  }

 private:
  /*! \brief gradient statistics */
  T grad_;
  /*! \brief second order gradient statistics */
  T hess_;

};
/*! \brief gradient statistics pair usually needed in gradient boosting */
using GradientPair = detail::GradientPairInternal<float>;

/*! \brief High precision gradient statistics pair */
using GradientPairPrecise = detail::GradientPairInternal<double>;
}// namespace detail






#endif //MINXGBOOST_BASE_H
