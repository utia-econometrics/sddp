#pragma once

#include <armadillo>
#include "ScenarioModel.h"

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


class AssetAllocationModel :
	public ScenarioModel
{
public:
	AssetAllocationModel(unsigned int stages, const vector<string> &assets, RiskParameters param, const colvec &mu, const mat &sigma);
	AssetAllocationModel(unsigned int stages, const vector<string> &assets, RiskParameters param, const mat &sample);
	virtual ~AssetAllocationModel(void);

	virtual void Evaluate(double *value) const;
	virtual Distribution* GetDistribution(unsigned int stage) const;
	virtual StageDependence GetStageDependence() const;
	virtual unsigned int GetDecisionSize(unsigned int stage) const;
	virtual double GetDiscountFactor(unsigned int stage) const;
	virtual double GetRecourseLowerBound(unsigned int stage) const;
	virtual double GetRecourseUpperBound(unsigned int stage) const;
	virtual void FillSubradient(unsigned int stage, const double *prev_decisions, const double *scenario, double objective, const double *duals, double &recourse, double *subgradient) const;
	virtual double CalculateUpperBound(unsigned int stage, const double *prev_decisions, const double *decisions, const double *scenario, double recourse_estimate, bool cut_tail) const;
	virtual void BuildCoinModel(CoinModelWrapper * coin_model, unsigned int stage, const double *prev_decisions, const double *scenario, vector<string> &decision_vars, vector<string> &dual_constraints) const;
	virtual double ApproximateDecisionValue(unsigned int stage, const double *prev_decisions, const double *scenario) const;
	virtual double GetTailAlpha(unsigned int stage) const;
	virtual double CalculateTailCutoff(unsigned int stage, double var_h) const;
#ifndef _DEBUG
	virtual void BuildCplexModel(IloEnv &env, IloModel &model, IloExpr &objective, unsigned int stage, const double *prev_decisions, const double *scenario, vector<IloNumVar> &decision_vars, vector<IloRange> &dual_constraints) const;
#endif // !_DEBUG


	colvec GetMu() const;
	mat GetSigma() const;

	vector<string> GetAssets() const {
		return assets_;
	}

	unsigned int GetAssetsCount() const {
		return assets_.size();
	}

	RiskParameters GetParam() const {
		return param_;
	}

protected:
	void Init(unsigned int stages, const vector<string> &assets, RiskParameters &param);
	double CalculateCapital(const double *decisions, const double *scenario) const;
	double CalculateNodeValue(const double *decisions) const;

	colvec mu_;
	mat sigma_;

	vector<string> assets_;
	RiskParameters param_;

	static double EPSILON;
};
