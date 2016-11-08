#include "ScenarioModel.h"
#include <boost/bind.hpp>
#include <boost/function.hpp>

using namespace boost;

double ScenarioModel::EPSILON = 0.000001;

ScenarioModel::ScenarioModel(unsigned int stages, const vector<string> &assets, RiskParameters param)
{
	stages_ = stages;
	assets_ = assets;
	param_ = param;
#ifdef _DEBUG
	if(param_.expectation_coefficients.size() != stages) {
		throw ScenarioModelException("Invalid size of expectation coefficients vector");
	}
	if(param_.risk_coefficients.size() != stages) {
		throw ScenarioModelException("Invalid size of risk coefficients vector");
	}

	//the coefficients for the first stage should be zero (deterministic!)
	if((stages > 0) && (abs(param_.expectation_coefficients[0]) > EPSILON)) {
		throw ScenarioModelException("Expectation coefficient for the first stage has to be zero");
	}
	if((stages > 0) && (abs(param_.risk_coefficients[0]) > EPSILON)) {
		throw ScenarioModelException("Risk coefficient for the first stage has to be zero");
	}
#endif
}

ScenarioModel::~ScenarioModel(void)
{
}


ScenarioTree* ScenarioModel::GetScenarioTree(vector<unsigned int> stage_samples) const {
	//get the distributons
	vector<Distribution *> distributions;
	for(unsigned int stage = 1; stage <= GetStagesCount(); ++stage) {
		distributions.push_back(this->GetDistribution(stage));
	}

	//get the model evaluation function
	boost::function<void(double *)> evaluate = boost::bind(&ScenarioModel::Evaluate, this, _1);

	//get the tree
	ScenarioTree *tree = new ScenarioTree(stages_, stage_samples, this->GetAssetsCount(), this->GetStageDependence(), distributions, evaluate);
	tree->GenerateTree();

	//clean the distributions
	for(unsigned int i = 0; i < distributions.size(); ++i) {
		delete distributions[i];
	}

	return tree;
}

Distribution * ScenarioModel::GetTrueDistribution(unsigned int stage) const {
	throw ScenarioModelException("Distribution not known");
}
