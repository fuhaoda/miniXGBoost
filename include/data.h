//
// Created by Haoda Fu on 2019-05-20.
//

#ifndef MINIXGBOOST_DATA_H
#define MINIXGBOOST_DATA_H

#include <iostream>
#include <vector>

namespace xgboost{
  namespace data{

    /*!
     * \brief Class for Data Entry
     */
    class Entry{
    public:
      size_t findex; //feature index
      float fvalue; //feature value

      //constructors
      Entry(){}
      Entry(size_t findex, float fvalue) : findex(findex), fvalue(fvalue){}

      /*! \brief compare fvalue for sorting the entry in a vector */
      inline static bool cmp_fvalue( const Entry &a, const Entry &b ){
        return a.fvalue < b.fvalue;
      }
    }; //class Entry


    class SimpleSparseMatrix{
    public:
      SimpleSparseMatrix(){ clear(); }

      inline void clear();
      /*!
      * \brief add a row to the matrix, with data stored in STL container
      * \param findex feature index
      * \param fvalue feature value
      * \return the row id added line
      */
      size_t addRow(const std::vector<size_t> & findex, const std::vector<float> & fvalue);

      void loadLibSVM(const std::string & dataFileName);
      void loadCSV(const std::string & dataFileName);

    private:

      std::vector<float> y_;
      //for row major sparse matrix
      std::vector<size_t>  row_ptr_;
      std::vector<Entry>  row_data_;

      //for column major sparse matrix
      std::vector<size_t>  col_ptr_;
      std::vector<Entry>  col_data_;
    }; //class SimpleSparseMatrix

  } //namespace data
} // namespace xgboost




#endif //MINIXGBOOST_DATA_H
