#pragma once
#include "ScenarioModel.h"

class CorrelatedNormal :
	public ScenarioModel
{
public:
	CorrelatedNormal(unsigned int stages, const vector<string> &assets, RiskParameters param, const colvec &mu, const mat &sigma);
	virtual ~CorrelatedNormal(void);

	virtual void Evaluate(double *value) const;
	virtual Distribution* GetDistribution(unsigned int stage) const;
	virtual StageDependence GetStageDependence() const;
	virtual Distribution* GetTrueDistribution(unsigned int stage) const;

protected:
	colvec mu_;
	mat sigma_;

	//counted in the beggining
	colvec mu1_;
};
