#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include "Exception.h"
using namespace std;

//runtime exception
class GamsWrapperException : Exception
{ 
public:
	GamsWrapperException(string text) : Exception(text){
	}
};


class GamsWrapper
{
public:
	GamsWrapper(const string &gamsPath);
	~GamsWrapper(void);
	string RunGams(const string &scriptName);
private:
	string gamsPath_;
	static const string GAMS_TMP_OUT;
};
