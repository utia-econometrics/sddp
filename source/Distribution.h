#pragma once

#include <string>
#include <map>
#include <iomanip>
#include <armadillo>
#include "RandomGenerator.h"

using namespace std;
using namespace arma;

enum DistributionType {
	DISTRIBUTION_NORMAL = 0,
	DISTRIBUTION_DISCRETE = 1,
	DISTRIBUTION_LOGNORMAL = 2
};

///zakladni trida pro rozdeleni, implementuje nektere spolecne veci pro ruzna rozdeleni
class DistributionException
{ 
public:
	DistributionException(string text) {
		text_ = text;
	}
	string what() const
	{ 
		return text_;
	}
private:
	string text_;
};

class Distribution
{
public:
	Distribution(const colvec &mu, const mat &sigma);
	Distribution(const mat &sample);
	virtual ~Distribution(void);

	//estimators
	virtual void UnbiasedEstimate();
	virtual double GetAlpha() const; //for elliptic distributions, the variance multiplier
	virtual void MLEstimate();
	
	//functions for moments of each marginal component
	virtual void SampleMean(rowvec & s_mean);
	virtual void SampleStdDev(rowvec & s_stddev);
	virtual void SampleSkewness(rowvec & s_skewness);
	virtual void SampleKurtosis(rowvec & s_kurtosis);

	//generate some sample
	virtual void GenerateAnalytic(mat &sample, int count) = 0;

	//return the type of distribution
	virtual DistributionType GetType() const = 0;

	//pdf & inverse
	virtual double DistributionFunction(double x) const = 0;
	virtual double InverseDistributionFunction(double y) const = 0;

    //Dimension of the data
	virtual unsigned int GetDimension() const = 0;
	
    //getters
	colvec GetMu() const {
		return mu_;
	}

	mat GetSigma() const {
		return sigma_;
	}


protected:
	///distirbution parameters
	colvec mu_;
	mat sigma_;

	///sample to estimate from
	mat sample_;

	///internal use only, correlation matrix, not the distribution parameters
	mat corr_;

	///generator of the U(0,1) distribution
	RandomGenerator* generator_;
};
