//
// Created by Haoda Fu on 2019-05-19.
//

#ifndef MINIXGBOOST_CONFIG_H
#define MINIXGBOOST_CONFIG_H

#include <iostream>
#include <fstream>
#include <vector>
#include "base.h"

namespace xgboost{
namespace common{
  class ConfigParse{
    public:
      explicit ConfigParse(const std::string & cfgFileName){
        fi_.open(cfgFileName);
        utils::myAssert(!fi_.fail(),"Cannot open configuration file!");
      }

      std::vector<std::pair<std::string, std::string> > parse(){
        std::vector<std::pair<std::string, std::string> > results{};
        char delimiter = '=';
        char comment = '#';
        std::string line{};
        std::string name{};
        std::string value{};

        while( !fi_.eof() ) {
          getline( fi_, line); //read a line of configure file
          line=line.substr(0,line.find(comment)); //anything beyond # is comment
          size_t delimiterPos = line.find(delimiter); //find the = sign
          name = line.substr(0,delimiterPos); //anything before = is the name
          value = line.substr(delimiterPos+1, line.length()-delimiterPos-1); //after this = is the value

          if(line.empty()||name.empty()||value.empty()) continue; //skip a line if # at beginning or there is no value or no name.
          cleanString(name); //clean the string
          cleanString(value);
          results.emplace_back(name,value);
        }

        return results;
      }

      ~ConfigParse(){
        fi_.close();
      }

    private:
      std::ifstream fi_;
      std::string allowableChar="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_./";
      void cleanString(std::string & str){
        size_t firstIndx = str.find_first_of(allowableChar);
        size_t lastIndx = str.find_last_of(allowableChar);
        str = str.substr(firstIndx,lastIndx-firstIndx+1); //this line can be more efficient, but keep as is for simplicity.
      }
    };
}
} //namespace xgboost
#endif //MINIXGBOOST_CONFIG_H
