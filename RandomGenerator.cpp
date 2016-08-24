#include "RandomGenerator.h"
#include "time.h"

bool RandomGenerator::init_ = true;
int RandomGenerator::seed_ = 0;

RandomGenerator::RandomGenerator(RngStream *generator)
{
	generator_ = generator;
}

RandomGenerator::~RandomGenerator(void)
{
	delete generator_;
}

RandomGenerator* RandomGenerator::GetGenerator() {

	if(init_) {
		//add some randomization
		unsigned long seed_arr[6];
		int seed = seed_;
		if(seed == 0) {
			//seed was not set
#ifndef _DEBUG
			seed = time(NULL);
#endif
		}
		for(int i = 0; i < 6; ++i) {
			seed_arr[i] = seed;
		}
		RngStream::SetPackageSeed (seed_arr);
		init_ = false;
	}

	//2 new objects, one gets deleted by generator itself, second one is up to generator user
	return new RandomGenerator(new RngStream());
}

double RandomGenerator::GetRandom() {
	return generator_->RandU01();
}

int RandomGenerator::GetRandomInt(int i, int j) {
#ifdef _DEBUG
	if(i > j) {
		throw RandomGeneratorException("Invalid bounds");
	}
#endif
	return generator_->RandInt(i, j);
}

SCENINDEX RandomGenerator::GetRandomHuge(SCENINDEX i, SCENINDEX j) {
#ifdef _DEBUG
	if(i > j) {
		throw RandomGeneratorException("Invalid bounds");
	}
	if(j - i > ULONG_MAX) {
		throw RandomGeneratorException("Random sampling from huge number is not yet implenented");
	}
#endif
	unsigned long dif = (j - i).convert_to<unsigned long>();
	unsigned long gen = static_cast<unsigned long>((dif + 1.0) * generator_->RandU01());
	return i + gen;
}