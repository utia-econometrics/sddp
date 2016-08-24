#include "CorrelatedNormal.h"
#include "NormalDistribution.h"

CorrelatedNormal::CorrelatedNormal(unsigned int stages, const vector<string> &assets, RiskParameters param, const colvec &mu, const mat &sigma)
: ScenarioModel(stages, assets, param)
{
	mu_ = mu;
	sigma_ = sigma;

	mu1_ = mu_ + ones(GetAssetsCount());
}

CorrelatedNormal::~CorrelatedNormal(void)
{
}

Distribution* CorrelatedNormal::GetDistribution(unsigned int stage) const {
	if(stage == 1) {
		//fixed root
		colvec mu = ones(GetAssetsCount());
		mat sigma = zeros(GetAssetsCount(), GetAssetsCount());
		return new NormalDistribution(mu, sigma);
	}
	return new NormalDistribution(mu1_, sigma_);
}

void CorrelatedNormal::Evaluate(double *value) const {
	//nothing to do
}

StageDependence CorrelatedNormal::GetStageDependence() const {
	return STAGE_INDEPENDENT;
}

Distribution* CorrelatedNormal::GetTrueDistribution(unsigned int stage) const {
	return GetDistribution(stage);
}
