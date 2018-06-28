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

unsigned int ScenarioModel::GetStatesCountStage(unsigned int stage) const
{
	return 1; //default
}

mat ScenarioModel::GetTransitionProbabilities(unsigned int stage) const
{
	return ones(1); //default one state and full transition
}

/* TODO: reconsider stage_samples to be fully general */
ScenarioTree* ScenarioModel::GetScenarioTree(vector<unsigned int> stage_samples) const {
	if (GetStageDependence() == STAGE_INDEPENDENT)
	{
		//get the distributons
		vector<Distribution *> distributions;
		for (unsigned int stage = 1; stage <= GetStagesCount(); ++stage) {
			distributions.push_back(this->GetDistribution(stage));
		}

		//get the model evaluation function to post process samples
		boost::function<void(double *)> evaluate = boost::bind(&ScenarioModel::Evaluate, this, _1);

		//get the tree
		ScenarioTree *tree = new ScenarioTree(stages_, stage_samples, GetStageDependence(), distributions, evaluate);
		tree->GenerateTree();

		//clean the distributions
		for (unsigned int i = 0; i < distributions.size(); ++i) {
			delete distributions[i];
		}

		return tree;
	}
	else if (GetStageDependence() == MARKOV) {
		//get the distributons
		vector<vector<Distribution *> > distributions;
		vector<unsigned int> state_counts;
		vector<mat> transition_probabilities;
		vector<vector<unsigned int> > all_stage_samples;
		for (unsigned int stage = 1; stage <= GetStagesCount(); ++stage) {
			unsigned int stat_cnt = GetStatesCountStage(stage);
			state_counts.push_back(stat_cnt);
			vector<Distribution*> one_stage_distr;
			vector<unsigned int> one_stage_samples;
			for (unsigned int state = 1; state <= stat_cnt; ++state) {
				one_stage_distr.push_back(GetDistribution(stage, state));
				one_stage_samples.push_back(stage_samples[stage - 1]);
			}
			distributions.push_back(one_stage_distr);
			all_stage_samples.push_back(one_stage_samples);
		}

		for (unsigned int stage = 1; stage < GetStagesCount(); ++stage) { //not done for last stage, no transition from it
			transition_probabilities.push_back(GetTransitionProbabilities(stage));
		}

		//get the model evaluation function to post process samples
		boost::function<void(double *)> evaluate = boost::bind(&ScenarioModel::Evaluate, this, _1);

		//get the tree
		ScenarioTree *tree = new ScenarioTree(stages_, state_counts, all_stage_samples, GetStageDependence(), distributions, transition_probabilities, evaluate);
		tree->GenerateTree();

		//clean the distributions
		for (unsigned int i = 0; i < distributions.size(); ++i) {
			for (unsigned int j = 0; j < distributions[i].size(); ++j) {
				delete distributions[i][j];
			}
		}

		return tree;
	}
	else {
		throw ScenarioModelException("Unsupported dependence model");
	}
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
