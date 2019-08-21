#pragma once

#include "Distribution.h"

class DiscreteDistribution :
	public Distribution
{
public:
	DiscreteDistribution(const mat &sample);
	virtual ~DiscreteDistribution(void);
	virtual void GenerateAnalytic(mat &sample, int count);
	virtual double DistributionFunction(double x) const;
	virtual double InverseDistributionFunction(double y) const;
	virtual DistributionType GetType() const;
	virtual unsigned int GetDimension() const;
};
