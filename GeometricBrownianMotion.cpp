#include "GeometricBrownianMotion.h"
#include "NormalDistribution.h"

#define DEBUG_DISTRIBUTION 0

GeometricBrownianMotion::GeometricBrownianMotion(unsigned int stages, const vector<string> &assets, RiskParameters param, const colvec &mu, const mat &sigma)
: ScenarioModel(stages, assets, param)
{
	mu_ = mu;
	sigma_ = sigma;
}

GeometricBrownianMotion::GeometricBrownianMotion(unsigned int stages, const vector<string> &assets, RiskParameters param, const mat &sample)
: ScenarioModel(stages, assets, param)
{
	mat estimator = log(sample);
	NormalDistribution normal(estimator);
	normal.MLEstimate();
	
	mu_ = normal.GetMu();
	sigma_ = normal.GetSigma();
}

GeometricBrownianMotion::~GeometricBrownianMotion(void)
{
}

colvec GeometricBrownianMotion::GetMu() const {
	return mu_;
}

mat GeometricBrownianMotion::GetSigma() const {
	return sigma_;
}

Distribution* GeometricBrownianMotion::GetDistribution(unsigned int stage) const {
#if DEBUG_DISTRIBUTION == 1
	if(true) {
#else
	if(stage == 1) {
#endif
		//fixed root e^0 = 1
		colvec mu = zeros(GetAssetsCount());
		mat sigma = zeros(GetAssetsCount(), GetAssetsCount());
		return new NormalDistribution(mu, sigma);
	}
	return new NormalDistribution(mu_, sigma_);
}

void GeometricBrownianMotion::Evaluate(double *value) const {
	colvec cur_value(value, GetAssetsCount(), false);
	cur_value = exp(cur_value);
}

StageDependence GeometricBrownianMotion::GetStageDependence() const {
	return STAGE_INDEPENDENT;
}