#pragma once

#include "Solver.h"
#include "boost/date_time/posix_time/posix_time.hpp"
#include "CoinModelWrapper.h"
#include <algorithm>
#include <ilcplex/ilocplex.h>
#include <boost/filesystem/fstream.hpp>

using namespace boost::posix_time;

enum ExternalSolver {
	SOLVER_COINOR,
	SOLVER_CPLEX
};

enum SddpStrategy {
	STRATEGY_DEFAULT,
	STRATEGY_CONDITIONAL
};

enum SddpSolverNodeType {
	NODE_DEFAULT,
	NODE_EXPECTATION,
	NODE_CVAR
};

struct SddpSolverCut {
	SddpSolverCut() {
		gradient = 0;
		gradient_size = 0;
	}
	SddpSolverCut(const SddpSolverCut& cut)
	{
		gradient_size = cut.gradient_size;
		absolute = cut.absolute;
		gradient = new double[gradient_size];
		copy(cut.gradient, cut.gradient + cut.gradient_size, gradient);
	}

	~SddpSolverCut() {
		if(gradient != 0) {
			delete[] gradient;
		}
	}

	double *gradient; //cuts for recourse function .. coefficients for x1, .. xN, var
	unsigned int gradient_size; //number of elements in the gradient
	double absolute; //absolute term for the cut
};

struct SddpSolverNode {
	SddpSolverNode() {
		parent = 0;
		solution = 0;
		subgradient = 0;
		type = NODE_DEFAULT;
		solved_forward_nr = 0;
		solved_backward_nr = 0;
		cut_added_nr = 0;
		cut_tail = false;
		var_subgradient = 0;
		var_subgradient_other = 0;
	}

	~SddpSolverNode() {
		if(solution != 0) {
			delete[] solution;
		}
		if(subgradient != 0) {
			delete[] subgradient;
		}
	}

	double GetValue() {
		double sum = 0.0;
		for(unsigned int i = 0; i < tree_node.GetSize(); ++i) {
			sum += solution[i];
		}
		return sum;
	}

	double GetCapital(const double *weights) {
		double sum = 0.0;
		const double *values = tree_node.GetValues();
		for(unsigned int i = 0; i < tree_node.GetSize(); ++i) {
			sum += weights[i] * values[i];
		}
		return sum;
	}

	double GetProbability() {
		return tree_node.GetProbability();
	}

	TreeNode tree_node; //equivalent node in the scenario tree
	SddpSolverNode *parent; //parent node or 0
	vector<SddpSolverNode *> descendants; //descendant nodes
	double *solution; //optimal weights so far
	double var; //current Value at Risk for calculating CVaR
	double var_subgradient; //subgradient of the VaR variable
	double value; //current value (for upper bound)
	double objective; //current objective value (lower bound)
	double *subgradient; //subgradient of current solution 
	unsigned int solved_forward_nr; //number of iteration when the model was solved
	unsigned int solved_backward_nr; //number of iteration when the model was solved
	unsigned int cut_added_nr; //number of iteration when we processed the cuts
	SddpSolverNodeType type; //type of the node when employing conditional sampling / path
	bool cut_tail; //if we can cut off the tail values for CVaR
	double var_other; //current Value at Risk for calculating CVaR
	double var_subgradient_other; //subgradient of the VaR variable
};

struct SddpComparableNode {
	SddpComparableNode(SCENINDEX c_index, double c_value, double c_probability) {
		index = c_index;
		value = c_value;
		probability = c_probability;
	}

	SCENINDEX index;
	double value;
	double probability;

	bool operator < (const SddpComparableNode& comp_node) const
    {
		return (value < comp_node.value);
    }
};

//runtime exception
class SddpSolverException : Exception
{ 
public:
	SddpSolverException(string text) : Exception(text){
	}
};

class SddpSolver :
	public Solver
{
public:
	SddpSolver(const ScenarioModel *model, unsigned int fix_descendant_count = 0, unsigned int fix_reduced_count = 0);
	virtual ~SddpSolver(void);

	virtual void Solve(mat &weights, double &objective);
	void Solve(mat &weights, double &lower_bound_exact, double &upper_bound_mean, double &upper_bound_bound);
	virtual void GetStageSamples(vector<unsigned int> &stage_samples);
	virtual void GetReducedSamples(vector<unsigned int> &stage_samples);

	void EvaluatePolicy(boost::function<vector<vector<double> >(vector<const double *>)> policy, double &return_mean, double &return_upper_bound, unsigned int iterations = 10);
	vector<vector<double> > GetPolicy(vector<const double *> scenario);

	SddpSolverNode *GetRoot();
	void ClearNodes();

protected:
	double GetRecourseLowerBound(unsigned int stage);
	double GetRecourseUpperBound(unsigned int stage);
	void BuildSolverTree();
	void DestroySolverTree();
	void SolveNode(SddpSolverNode *node);
	void SolveNodeCplex(SddpSolverNode *node);
	void SolveNodeCoinOr(SddpSolverNode *node);
	void AddCut(SddpSolverNode *node);
	void ForwardPassStandard(unsigned int count, vector<SCENINDEX> &nodes, bool solve = true);
	void ForwardPassConditional(unsigned int count, vector<SCENINDEX> &nodes, bool solve = true);
	void BackwardPass(const vector<SCENINDEX> &nodes);
	SddpSolverNode *BuildNode(TreeNode treenode);
	SddpSolverNode *BuildNode(SCENINDEX index);
	bool NodeExists(TreeNode treenode);
	bool NodeExists(SCENINDEX index);
	void CalculateTotalValue(SddpSolverNode *node, double &value, double &probability);
	double CalculateNodeCapital(SddpSolverNode *node);
	double CalculateNodeCapital(SddpSolverNode *node, SddpSolverNode *parent);
	void ClearNodeDescendants(SddpSolverNode *node);
	void ClearSingleNode(SddpSolverNode *node);
	void ChooseRandomIndices(SCENINDEX start, SCENINDEX end, unsigned int count, vector<SCENINDEX> & indices);
	void ChooseRandomSubset(const vector<SCENINDEX> & set, unsigned int count, vector<SCENINDEX> & subset);
	void ConnectNode(SddpSolverNode *node, SddpSolverNode *parent);
	double CalculateUpperBoundDefault(const vector<SCENINDEX> &nodes);
	double CalculateUpperBoundConditional(const vector<SCENINDEX> &nodes);
	double GetNodeProbability(SddpSolverNode *node);
	double GetConditionalProbability(SddpSolverNode *node);
	double GetConditionalProbability(unsigned int stage);
	SddpSolverNode *SampleNodeByProbability(SddpSolverNode *parent, double probability);

	map<SCENINDEX, SddpSolverNode *> nodes_;
	vector<vector<SddpSolverCut> > cuts_;
	RandomGenerator *generator_;
	unsigned int forward_iterations_; //iteration nr of the algorithm
	unsigned int backward_iterations_; //iteration nr of the algorithm

	unsigned int solved_lps_; //number of LPs solved
	time_duration lp_solve_time_; //time spent in a LP solver
	
	unsigned int fix_descendant_count_; //fixes descendant count for each stage
	unsigned int fix_reduced_count_; //fixes reduced descendant count for each stage

	//constants
	const static double CONVERGENCE_BOUND;
	const static unsigned int FORWARD_COUNT;
	const static unsigned int BACKWARD_COUNT;
	const static unsigned int MAX_ITERATIONS;
	const static ExternalSolver EXTERNAL_SOLVER;
	const static SddpStrategy SOLVER_STRATEGY;
	const static double CONDITIONAL_PROBABILITY;
	const static double STOP_CONFIDENCE;
	const static int STOP_ELEMENTS;
	const static double EPSILON;
	const static bool COMPARISON_EQUAL;
	const static bool QUICK_CONDITIONAL;
	const static bool USE_IMPORTANCE;
	const static int FORWARD_FIXED_STAGE_NODES;

	//output files
	boost::filesystem::ofstream value_file_;
	boost::filesystem::ofstream value_file_forward_;
	map<SCENINDEX, bool> reported_forward_values_;

	//coefficients
	vector<double> risk_coefficients_;
	vector<double> risk_coefficients_other_;
	vector<double> expectation_coefficients_;
};
