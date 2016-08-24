#pragma once

#include <boost/lexical_cast.hpp>
#include <string>
#include <vector>

//type for the scenario index
#include <boost/multiprecision/cpp_int.hpp> 
#define SCENINDEX boost::multiprecision::uint128_t
using namespace std;

class Helper
{
public:
	Helper(void);
	~Helper(void);

	template<class T> static vector<string> ArrayToString(const vector<T> &data) {
		vector<string> vec;
		for(unsigned int i = 0; i < data.size(); ++i) {
			vec.push_back(lexical_cast<string>(data[i]));
		}
		return vec;
	}
};
