#pragma once

#include "ScenarioModel.h"

class GeometricBrownianMotion :
	public ScenarioModel
{
public:
	GeometricBrownianMotion(unsigned int stages, const vector<string> &assets, RiskParameters param, const colvec &mu, const mat &sigma);
	GeometricBrownianMotion(unsigned int stages, const vector<string> &assets, RiskParameters param, const mat &sample);
	virtual ~GeometricBrownianMotion(void);

	virtual void Evaluate(double *value) const;
	virtual Distribution* GetDistribution(unsigned int stage) const;
	virtual StageDependence GetStageDependence() const;

	colvec GetMu() const;
	mat GetSigma() const;

protected:
	colvec mu_;
	mat sigma_;
};
