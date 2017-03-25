#include "ScenarioModel.h"
#include <boost/bind.hpp>
#include <boost/function.hpp>

using namespace boost;

ScenarioModel::ScenarioModel(unsigned int stages)
{
	stages_ = stages;
}

ScenarioModel::~ScenarioModel(void)
{
}

void ScenarioModel::Evaluate(double * value) const
{
	//no post processing
}

StageDependence ScenarioModel::GetStageDependence() const
{
	return STAGE_INDEPENDENT; //default
}

ScenarioTree* ScenarioModel::GetScenarioTree(vector<unsigned int> stage_samples) const {
	//get the distributons
	vector<Distribution *> distributions;
	for(unsigned int stage = 1; stage <= GetStagesCount(); ++stage) {
		distributions.push_back(this->GetDistribution(stage));
	}

	//get the model evaluation function to post process samples
	boost::function<void(double *)> evaluate = boost::bind(&ScenarioModel::Evaluate, this, _1);

	//get the tree
	ScenarioTree *tree = new ScenarioTree(stages_, stage_samples, this->GetStageDependence(), distributions, evaluate);
	tree->GenerateTree();

	//clean the distributions
	for(unsigned int i = 0; i < distributions.size(); ++i) {
		delete distributions[i];
	}

	return tree;
}

double ScenarioModel::GetDiscountFactor(unsigned int stage) const {
	return 1.0; //default
}

Distribution * ScenarioModel::GetTrueDistribution(unsigned int stage) const {
	throw ScenarioModelException("Distribution not known");
}

double ScenarioModel::ApproximateDecisionValue(unsigned int stage, const double * prev_decisions, const double * scenario) const
{
	throw ScenarioModelException("Importance sampling not implemented for the model");
}

double ScenarioModel::GetTailAlpha(unsigned int stage) const
{
	throw ScenarioModelException("Importance sampling not implemented for the model");
}

double ScenarioModel::CalculateTailCutoff(unsigned int stage, double var_h) const
{
	throw ScenarioModelException("Importance sampling not implemented for the model");
}
