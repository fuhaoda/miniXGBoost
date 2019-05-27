//
// Created by Haoda Fu on 2019-05-20.
//

#ifndef MINIXGBOOST_DATA_H
#define MINIXGBOOST_DATA_H

#include <iostream>
#include <vector>
#include "base.h"

namespace xgboost{
  namespace data{

    /*!
     * \brief Class for Data Entry
     */
    class Entry{
    public:
      size_t findex{}; //feature index
      float fvalue{}; //feature value

      //constructors
      Entry()= default;
      Entry(size_t findex, float fvalue) : findex(findex), fvalue(fvalue){}

      /*! \brief compare fvalue for sorting the entry in a vector */
      inline static bool cmp_fvalue( const Entry &a, const Entry &b ){
        return a.fvalue < b.fvalue;
      }

      inline static bool cmp_findex(const Entry &a, const Entry &b){
        return a.findex < b.findex;
      }
    }; //class Entry


    class SimpleSparseMatrix{
    public:
      SimpleSparseMatrix(){ clear(); }

      void clear();
      /*!
      * \brief add a row to the matrix, with data stored in STL container
      * \param findex feature index
      * \param fvalue feature value
      * \return the row id added line
      */
      //load data
      void loadLibSVM(const std::string & dataFileName);
      void loadCSV(const std::string & dataFileName);

      //overall
      inline size_t numOfEntry() const;
      size_t sampleSize() const;
      const float overallResponseMean() const;
      const std::vector<float> & getY() const;

      //row operations
      size_t addRow(const std::vector<size_t> & findex, const std::vector<float> & fvalue);

      using RowIter = utils::ForwardIterator<Entry>;
      size_t numOfRow() const;
      RowIter getARow(size_t rowIndex) const; //RowIter is left open and right closed.

      //column operations
      void translateToCSCFormat(); //call this function after load data
      size_t numOfCol() const;
      using ColIter = utils::ForwardIterator<Entry>;
      using ColReverseIter=utils::BackwardIterator<Entry>;
      ColIter getACol(size_t colIndex) const;
      ColReverseIter getAColRevese(size_t colIndex) const;


    private:

      std::vector<float> y_;
      //for row major sparse matrix
      std::vector<size_t>  row_ptr_;
      std::vector<Entry>  row_data_;

      //for column major sparse matrix
      std::vector<size_t>  col_ptr_;
      std::vector<Entry>  col_data_;

      bool colAccess{false};
    }; //class SimpleSparseMatrix

  } //namespace data
} // namespace xgboost




#endif //MINIXGBOOST_DATA_H
