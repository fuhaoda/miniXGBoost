//
// Created by Haoda Fu on 2019-05-18.
//

#ifndef MINXGBOOST_BASE_H
#define MINXGBOOST_BASE_H

#include <iostream>

namespace xgboost {
  /*!
 * \brief unsigned integer type used in boost,
 *  used for feature index and row index.
 */
  using bst_uint = uint32_t;  // NOLINT
  using bst_int = int32_t;    // NOLINT
/*! \brief long integers */
  typedef uint64_t bst_ulong;  // NOLINT(*)
/*! \brief float type, used for storing statistics */
  using bst_float = float;  // NOLINT
  namespace detail {
    /*! \brief Implementation of gradient statistics pair. Template specialisation
    * may be used to overload different gradients types e.g. low precision, high
    * precision, integer, floating point. */
    template <typename T>
    class GradientPairInternal{

    public:
      //constructor
      GradientPairInternal(): grad_(0), hess_(0){}
      GradientPairInternal(float grad, float hess):grad_(grad), hess_(hess){}
      //copy constructor
      GradientPairInternal(const GradientPairInternal<T> & gpair):grad_(gpair.grad_), hess_(gpair.hess_){}

      void setGrad(float grad) {grad_=grad;}
      void setHess(float hess) {hess_=hess;}
      float getGrad() const{ return grad_;}
      float getHess() const{return hess_;}


      //define operator += and -=. This operation is preferred with the original copy is not needed.
      GradientPairInternal<T> &operator+=(const GradientPairInternal<T> & rhs){
        grad_+=rhs.grad_;
        hess_+=rhs.hess_;
        return *this;
      }

      GradientPairInternal<T> &operator-=(const GradientPairInternal<T> & rhs){
        grad_-=rhs.grad_;
        hess_-=rhs.hess_;
        return *this;
      }

      //define operator + and -. This is used when we need to assign the results to a new gradient pair.
      GradientPairInternal<T> &operator+(const GradientPairInternal<T> & rhs){
        GradientPairInternal<T> gpair;
        gpair.grad_=grad_+rhs.grad_;
        gpair.hess_=hess_+rhs.hess_;
        return gpair;
      }



      GradientPairInternal<T> &operator-(const GradientPairInternal<T> & rhs){
        GradientPairInternal<T> gpair;
        gpair.grad_=grad_-rhs.grad_;
        gpair.hess_=hess_-rhs.hess_;
        return gpair;
      }

    private:
      /*! \brief gradient statistics */
      T grad_;
      /*! \brief second order gradient statistics */
      T hess_;

    };

  }// namespace detail
  /*! \brief gradient statistics pair usually needed in gradient boosting */
  using GradientPair = detail::GradientPairInternal<float>;

  /*! \brief High precision gradient statistics pair */
  using GradientPairPrecise = detail::GradientPairInternal<double>;

  //utils
  namespace utils{

    inline void error(const std::string & msg){
      fprintf(stderr, "Error: %s\n", msg.c_str());
      exit(-1);
    }

    inline void myAssert(bool exp){
    if(!exp) error("Assert Error!");
  }

  inline void myAssert(bool exp, const std::string & msg){
      if (!exp) error(msg);
  }

  inline void warning(const std::string & msg){
    fprintf(stderr, "Warning: %s \n", msg.c_str());
  }

  inline void warning(bool exp, const std::string & msg){
      if(exp) fprintf(stderr, "Warning: %s \n", msg.c_str());
    }
  } //namespace utils
} //namespace xgboost

#endif //MINXGBOOST_BASE_H
