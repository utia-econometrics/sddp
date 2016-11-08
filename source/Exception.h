#pragma once

#include <string>
using namespace std;

class Exception
{
public:
	Exception(string text) {
		text_ = text;
	}
	virtual string what() const
	{ 
		return text_;
	}
protected:
	string text_;
};