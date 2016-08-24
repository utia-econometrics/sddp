#pragma once
#include "normaldistribution.h"
class LognormalDistribution :
	public NormalDistribution
{
public:
	LognormalDistribution(const colvec &mu, const mat &sigma);
	LognormalDistribution(const mat &sample);
	virtual ~LognormalDistribution(void);
	virtual void GenerateAnalytic(mat &sample, int count);
	virtual double DistributionFunction(double x) const;
	virtual double InverseDistributionFunction(double y) const;
	double SampleJarqueBera();
	virtual DistributionType GetType() const;
};

