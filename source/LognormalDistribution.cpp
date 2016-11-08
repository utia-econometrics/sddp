#include "LognormalDistribution.h"
#include "alglib/specialfunctions.h"
#include <math.h>

using namespace alglib;
using namespace arma;

LognormalDistribution::LognormalDistribution(const colvec &mu, const mat &sigma) : NormalDistribution(mu, sigma)
{

}

LognormalDistribution::LognormalDistribution(const mat &sample) : NormalDistribution(sample)
{

}

LognormalDistribution::~LognormalDistribution(void)
{

}


void LognormalDistribution::GenerateAnalytic(mat &sample, int count)
{
	NormalDistribution::GenerateAnalytic(sample, count);
	sample = exp(sample);
}

double LognormalDistribution::DistributionFunction(double x) const {
	if(mu_.n_rows != 1) {
		throw DistributionException("This function is for univariate normal distribution only.");
	}
	double mu = mu_(0);
	double sigma = sqrt(sigma_(0, 0));
	return normaldistribution( (log(x) - mu) / sigma);
}

double LognormalDistribution::InverseDistributionFunction(double y) const {
	if(mu_.n_rows != 1) {
		throw DistributionException("This function is for univariate normal distribution only.");
	}
	double mu = mu_(0);
	double sigma = sqrt(sigma_(0, 0));
	return exp(mu + sigma * invnormaldistribution(y));
}

double LognormalDistribution::SampleJarqueBera() {
	throw DistributionException("This test is for univariate normal distribution only.");
}

DistributionType LognormalDistribution::GetType() const {
	return DISTRIBUTION_LOGNORMAL;
}