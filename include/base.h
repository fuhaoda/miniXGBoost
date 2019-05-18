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
    class A{
    public:
      int a;
      int b;
      int add(int, int);
    };

  }
}
#endif //MINXGBOOST_BASE_H
