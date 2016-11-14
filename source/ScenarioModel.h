#pragma once

#include <armadillo>
#include "Distribution.h"
#include "ScenarioTree.h"

using namespace std;

enum RiskMeasure {
	RISK_CVAR_NESTED = 0,
	RISK_CVAR_SUM = 1,
	RISK_CVAR_MULTIPERIOD = 2,
	//more to be added
};

struct RiskParameters {
	double confidence;
	double discount_factor;
	vector<double> risk_coefficients;
	vector<double> expectation_coefficients;
	RiskMeasure risk_measure;
	double transaction_costs;
	double confidence_other;
	vector<double> risk_coefficients_other;
};


//runtime exception
class ScenarioModelException : Exception
{ 
public:
	ScenarioModelException(string text) : Exception(text){
	}
};

class ScenarioModel
{
public:
	ScenarioModel(unsigned int stages, const vector<string> &assets, RiskParameters param);
	virtual ~ScenarioModel(void);

	//converts the sampled value to the modeled process
	virtual void Evaluate(double *value) const = 0;

	//returns the distribution to be sampled at stage t
	virtual Distribution* GetDistribution(unsigned int stage) const = 0;

	//return the stage dependence in this model
	virtual StageDependence GetStageDependence() const = 0;

	//generates the scenario sampled tree
	virtual ScenarioTree* GetScenarioTree(vector<unsigned int> stage_samples) const;

	//returns the "true" distribution of the stage t values, if known
	virtual Distribution* GetTrueDistribution(unsigned int stage) const;

protected:
	vector<string> assets_;
	RiskParameters param_;
	unsigned int stages_;

	static double EPSILON;

public:
	unsigned int GetStagesCount() const {
		return stages_;
	}

	vector<string> GetAssets() const {
		return assets_;
	}

	unsigned int GetAssetsCount() const {
		return assets_.size();
	}

	RiskParameters GetParam() const {
		return param_;
	}

	vector<unsigned int> GetStages() const {
		vector<unsigned int> st;
		for(unsigned int i = 1; i <= stages_; ++i) {
			st.push_back(i);	
		}
		return st;
	}
};
