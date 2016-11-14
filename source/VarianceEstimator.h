#pragma once

#include "RandomGenerator.h"
#include <armadillo>

using namespace arma;

class VarianceEstimator
{
public:
	VarianceEstimator(void);
	~VarianceEstimator(void);

	void EstimateNormal(double alpha, double lambda, double probability, double &mean, double &variance);
	void EstimateLognormal(double alpha, double lambda, double probability, double &mean, double &variance);
	void EstimateDiscrete(colvec &scenarios, double alpha, double lambda, double probability, double &mean, double &variance);

protected:
	RandomGenerator *generator_;
};

