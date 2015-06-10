#pragma once
#include <string>
#include <unordered_set>
#include "Base.h"

class Simple: public Base
{
public:
  void CreateVocab(std::ifstream &corpusStrme);
  void CalcOOV(std::ifstream &testStrme);

protected:
  	std::unordered_set<std::string> vocab;


};