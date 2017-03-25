#pragma once

#include "Distribution.h"
#include "ScenarioTree.h"
#include "CoinModelWrapper.h"
#include <ilcplex/ilocplex.h>

using namespace std;

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
	//default constructor, takes number of stages
	ScenarioModel(unsigned int stages);
	virtual ~ScenarioModel(void);

	//converts the sampled value to the modeled process = sampling post processing
	virtual void Evaluate(double *value) const;

	//returns the distribution to be sampled at stage t
	virtual Distribution* GetDistribution(unsigned int stage) const = 0;

	//returns the size of the decision vector at stage t
	virtual unsigned int GetDecisionSize(unsigned int stage) const = 0;

	//return the stage dependence in this model, default is stage independent
	virtual StageDependence GetStageDependence() const;

	//generates the scenario sampled tree
	virtual ScenarioTree* GetScenarioTree(vector<unsigned int> stage_samples) const;

	//returns the "true" distribution of the stage t values, if known
	virtual Distribution* GetTrueDistribution(unsigned int stage) const;

	//returns initial lower bound for the recourse function for algorithm init
	virtual double GetRecourseLowerBound(unsigned int stage) const = 0;

	//returns initial upper bound for the recourse function for algorithm init
	virtual double GetRecourseUpperBound(unsigned int stage) const = 0;

	//returns discount factor for the stage t
	virtual double GetDiscountFactor(unsigned int stage) const;

	/* calculates an upper bound for a single node
	 stage - the current stage of the node
	 prev_decisions - current value of decisions from the previous stage
	 decisions - current optimal solutions of this stage
	 scenario - random parameters of this node
	 recourse estimate - an estimate of future recourse received from the solver
	 cut_tail - denotes possibility to not include CVaR tail terms into evaluation
	*/
	virtual double CalculateUpperBound(unsigned int stage, const double *prev_decisions, const double *decisions, const double *scenario, double recourse_estimate, bool cut_tail) const = 0;

	/* calculates a subgradient 
	inputs:
	 stage - the current stage of the node
	 prev_decisions - current value of decisions from the previous stage
	 scenario - random parameters of this node
	 objective - current objective function of this node
	 duals - dual solutions for this node
	outputs:
	 recourse - value of the recourse function at this node
	 subgradient - double vector to be filled in with subgradients for our decisions
	*/
	virtual void FillSubradient(unsigned int stage, const double *prev_decisions, const double *scenario, double objective, const double *duals, double &recourse, double *subgradient) const = 0;

	/* builds a model for COIN-OR solver
	 coin_model - model is to be built using the wrapper which is pre initialized and sent as paramater here
	 stage - the current stage of the node
	 prev_decisions - current value of decisions from the previous stage
	 scenario - random parameters of this node
	outputs:
	 decision_vars - vector of names of decision variables to be handled by the solver
	 dual_constraints - vector of names of constraints which should be analyzed for dual values by the solver
	*/
	virtual void BuildCoinModel(CoinModelWrapper * coin_model, unsigned int stage, const double *prev_decisions, const double *scenario, vector<string> &decision_vars, vector<string> &dual_constraints) const = 0;

#ifndef _DEBUG
	/* builds a model for CPLEX solver
	env - environment wrapper which is pre initialized and sent as paramater here
	model - model wrapper which is pre initialized and sent as paramater here
	objective - objective wrapper which is pre initialized and sent as paramater here
	stage - the current stage of the node
	prev_decisions - current value of decisions from the previous stage
	scenario - random parameters of this node
	outputs:
	decision_vars - vector of names of decision variables to be handled by the solver
	dual_constraints - vector of names of constraints which should be analyzed for dual values by the solver
	*/
	virtual void BuildCplexModel(IloEnv &env, IloModel &model, IloExpr &objective, unsigned int stage, const double *prev_decisions, const double *scenario, vector<IloNumVar> &decision_vars, vector<IloRange> &dual_constraints) const = 0;
#endif

	/* calculates approximate value of the decision from stage t - 1 at stage t .. 
	   Approximation function "h" from Kozmik,Morton(2015)
	  stage - the current stage of the node
	  prev_decisions - current value of decisions from the previous stage
	  scenario - random parameters of this node
	*/
	virtual double ApproximateDecisionValue(unsigned int stage, const double *prev_decisions, const double *scenario) const;

	/* defines the confidence level alpha to compute "VaR[h]" from Kozmik,Morton(2015)
	 stage - the current stage of the node
	*/
	virtual double GetTailAlpha(unsigned int stage) const;

	/* defines a cutoff for node being in tail part involved in CVaR calculation .. 
	   Margin function "m" from Kozmik,Morton(2015)
	 stage - the current stage of the node
	 parameter var_h - Value at Risk of the ApproximateDecisionValue "h"
	*/
	virtual double CalculateTailCutoff(unsigned int stage, double var_h) const;

protected:
	unsigned int stages_;

public:
	//returns total count of stages
	unsigned int GetStagesCount() const {
		return stages_;
	}

	//reutrns vector with stage numbers
	vector<unsigned int> GetStages() const {
		vector<unsigned int> st;
		for(unsigned int i = 1; i <= stages_; ++i) {
			st.push_back(i);	
		}
		return st;
	}
};
