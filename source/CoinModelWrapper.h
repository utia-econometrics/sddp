#pragma once

#include "OsiCbcSolverInterface.hpp"
#include "OsiClpSolverInterface.hpp"
#include "CoinPackedMatrix.hpp"
#include "CoinPackedVector.hpp"

#include <map>
#include "Exception.h"

using namespace std;

enum BoundType {
	LOWER_THAN,
	EQUAL_TO,
	GREATER_THAN
};

//runtime exception
class CoinModelWrapperException : Exception
{ 
public:
	CoinModelWrapperException(string text) : Exception(text){
	}
};

class CoinModelWrapper
{
public:
	CoinModelWrapper(void);
	~CoinModelWrapper(void);

	void AddVariable(string variable);
	void AddLowerBound(string variable, double bound);
	void AddUpperBound(string variable, double bound);
	void AddObjectiveCoefficient(string variable, double value);
	void AddConstraint(string constraint);
	void AddConstraintVariable(string constraint, string variable, double value);
	void AddConstrainBound(string constraint, BoundType bound_type, double value);
	void Solve();
	double GetSolution(string variable);
	double GetObjective();
	double GetDualPrice(string constraint);
	
	friend ostream& operator<<(ostream& os, const CoinModelWrapper& cmw);
protected:
	inline void AssertValidVariable(string variable) const;
	inline void AssertValidConstraint(string constraint) const;
	inline void AssertModelSolved() const;
	void Fill_array(double *arr, map<string, bool> &indices, map<string, double> &values, double def_value) const;
	void BuildModel();
	void ProcessSolution();

	map<string, bool> variables_;
	map<string, bool> constraints_;
	map<string, double> variable_upper_bounds_;
	map<string, double> variable_lower_bounds_;
	map<string, double> variable_objective_coefs_;
	map<string, double> constraint_upper_bounds_;
	map<string, double> constraint_lower_bounds_;
	map<string, map<string, double> *> coinstraint_variables_;
	OsiCbcSolverInterface* solver_interface_;
	bool model_built_;
	bool model_solved_;
	double objective_value_; //solved objective
	map<string, double> variable_solutions_;
	map<string, double> dual_prices_;
};
