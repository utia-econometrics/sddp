#pragma once

#include "Distribution.h"

class NormalDistribution :
	public Distribution
{
public:
	NormalDistribution(const colvec &mu, const mat &sigma);
	NormalDistribution(const mat &sample);
	virtual ~NormalDistribution(void);
	virtual void GenerateAnalytic(mat &sample, int count);
	virtual double DistributionFunction(double x) const;
	virtual double InverseDistributionFunction(double y) const;
	double SampleJarqueBera();
	virtual DistributionType GetType() const;

protected:
	double GenerateStandardized();

	bool spare_sample_ready_;
	double spare_sample_;
};
