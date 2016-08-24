#pragma once

#include "rngstream/RngStream.h"
#include "Helper.h"
#include "Exception.h"

//runtime exception
class RandomGeneratorException : Exception
{ 
public:
	RandomGeneratorException(string text) : Exception(text){
	}
};

class RandomGenerator
{

public:
	static RandomGenerator* GetGenerator(void);
	~RandomGenerator(void);
	double GetRandom();
	int GetRandomInt(int i, int j);
	SCENINDEX GetRandomHuge(SCENINDEX i, SCENINDEX j);
	
	static void SetSeed(int seed) {
		seed_ = seed;
	}

protected:
	RandomGenerator(RngStream *generator);

	//actual random generator interface
	RngStream *generator_;

	//should we randomly init the generator?
	static bool init_;

	//seed provided for initialization
	static int seed_;
};
