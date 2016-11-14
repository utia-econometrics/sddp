#pragma once

#include "Exception.h"
#include "ScenarioTree.h"
#include <armadillo>
#include "ScenarioModel.h"
#include <string>
#include <vector>

using namespace std;
using namespace arma;

//runtime exception
class SolverException : Exception
{ 
public:
	SolverException(string text) : Exception(text){
	}
};

class Solver
{
public:
	Solver(const ScenarioModel *model);
	virtual ~Solver(void);

	//solves the specified model
	virtual void Solve(mat &weights, double &objective) = 0;

	//prescribes number of samples needed in each stage to run the solver
	virtual void GetStageSamples(vector<unsigned int> &stage_samples) = 0;

	//scenario reduction
	virtual void GetReducedSamples(vector<unsigned int> &stage_samples);

	virtual ScenarioTree* GetTree(bool regenerate = false);

	virtual void ForceTree(ScenarioTree * tree);

protected:

	virtual void Clean();

	const ScenarioModel *model_;
	ScenarioTree *tree_;
	bool tree_forced_;
};
