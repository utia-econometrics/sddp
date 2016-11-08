#pragma once

#include "Solver.h"

//runtime exception
class SaaNestedSolverException : Exception
{ 
public:
	SaaNestedSolverException(string text) : Exception(text){
	}
};

class SaaNestedSolver: public Solver
{
public:
	SaaNestedSolver(const ScenarioModel *model);
	~SaaNestedSolver(void);

	virtual void Solve(mat &weights, double &objective);

	virtual void GetStageSamples(vector<unsigned int> &stage_samples);

protected:	
	ScenarioTree *tree_;

	//generate GAMS scripts
	string SetsGams();
	string ParametersGams();
	string VariablesGams();
	string EquationsGams();
	string SolveGams();

	//read GAMS output
	void ParseOutputGams(mat &weights, double &objective);

	//GAMS PUTFILE path and EXE path
	static const string GAMS_VAR_OUT;
	static const string GAMS_EXE_PATH;
	static const string GAMS_SCRIPT;
	static const string GAMS_RISK_VAR;
	static const string GAMS_PROFIT_VAR;
	static const string GAMS_MODEL_VAR;
	static const string GAMS_SOLVER_VAR;
	static const string GAMS_OBJECTIVE_VAR;
};
