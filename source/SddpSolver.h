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
		dimension = 0;
		solution = 0;
		subgradient = 0;
		type = NODE_DEFAULT;
		solved_forward_nr = 0;
		solved_backward_nr = 0;
		cut_tail = false;
	}

	~SddpSolverNode() {
		if(solution != 0) {
			delete[] solution;
		}
		if(subgradient != 0) {
			delete[] subgradient;
		}
	}

	double GetProbability() {
		return tree_node.GetProbability();
	}

	unsigned int GetStage() {
		return tree_node.GetStage();
	}

	unsigned int GetState() {
		return tree_node.GetState();
	}

	TreeNode tree_node; //equivalent node in the scenario tree
	SddpSolverNode *parent; //parent node or 0
	vector<SddpSolverNode *> descendants; //descendant nodes
	unsigned int dimension; //size of the solution and subgradient vectors
	double *solution; //optimal weights so far
	double recourse; //recourse value of the node - usually == objective, but there might be some transformation
	double *subgradient; //subgradient of current solution 
	double recourse_value; //current  recourse value (for upper bound)
	double objective; //current objective value (lower bound)
	unsigned int solved_forward_nr; //number of iteration when the model was solved
	unsigned int solved_backward_nr; //number of iteration when the model was solved
	vector<unsigned int> cut_added_nr; //number of iteration when we processed the cuts, indexed by states
	SddpSolverNodeType type; //type of the node when employing conditional sampling / path
	bool cut_tail; //if we can cut off the tail values for CVaR
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
class SddpSolverException : public Exception
{ 
public:
	SddpSolverException(string text) : Exception(text){
	}
};

//algorithm config
struct SddpSolverConfig {
	//defined how many nodes should be sampled per each stage, 0 means solver default
	unsigned int samples_per_stage = 0;

	//defines to how many nodes per stage reduction shall be done, 0 = no scenario reduction performed
	unsigned int reduced_samples_per_stage = 0;


	//defines how many nodes to sample in each forward pass
	unsigned int forward_count = 10000;

	/*
	defines how many of the bacward pass nodes to select in and compute cuts for
	this needs to be lower or equal to the forward_count
	*/
	unsigned int backward_count = 25;

	//defines maximum number of iterations for the algorithm, 0 = unlimited
	unsigned int max_iterations = 200;

#ifdef _DEBUG
	ExternalSolver external_solver = SOLVER_COINOR;
#else
	ExternalSolver external_solver = SOLVER_CPLEX;
#endif

	//defines absolute acceptable difference between lower and upper bound to stop
	double convergence_bound = 0.02;

	//use statistical tests to determine stopping upper bound, if turned off, mean is used instead
	bool stop_use_tests = true;
	//defines confidence level of testing the difference between lower and upper bound
	double stop_confidence = 0.95;

	//defines after how many same lower bound algorithm decides to stop -> no more improving
	unsigned int stop_elements = 10;
	//defines negligible difference for testing of lower bound not imroving
	double epsilon = 1e-8;

	//choose between standard strategy and importance sampling
	SddpStrategy solver_strategy = STRATEGY_DEFAULT; //STRATEGY_CONDITIONAL;

	//noder per stage in the forward pass estimators
	unsigned int forward_fixed_stage_nodes = 0;


	//defines the probability to select tail node in importance sampling (CVaR)
	double conditional_probability = 0.3;

	//enables reduction of upper bound bias if certain conditions are met
	bool cut_nodes_not_tail = false;

	/*
	heuristic optimize the nodes selection in first stages .. implementation not ideally finished for importance/conditional
	needs to be turned off for inequal probabilities of nodes!!
	*/
	bool quick_conditional = false;


	//defines that the algorithm shall stop after N cuts have been collected. 0 = unlimited
	unsigned int stop_cuts = 0;

	//defines if to output detailed statistics about each iteration
	bool report_computation_times = false;

	//enables output of the node values into a textfile
	bool report_node_values = false;

	//enables calculatetion of solutions for all stages (averaging)
	bool calculate_future_solutions = false;
	unsigned int calculate_future_solutions_count = 10000;

	//Defines debugging of the forward pass - always selects all descendants (exponentially hard)
	bool debug_forward = false;

	//Defines debugging of the scenario tree - ovverides scenario count and generates only one descendant per stage
	bool debug_tree = false;

	/*
	Enables upper bound debugging - after stabilization of solution, the algorithm performs series 
	of upper bound computations (user can set their number and count of nodes in each iteration)
	Basic statistics are returned afterwards
	*/
	bool debug_bound = false;
	//how many upper bounds to comute
        unsigned int debug_bound_count = 100;
	//how many nodes to include in each iteration
	unsigned int debug_bound_nodes = 10000;

	//enables output of the LP model from the solver and debug info
	bool debug_solver = false;
};

class SddpSolver :
	public Solver
{
public:
	SddpSolver(const ScenarioModel *model, SddpSolverConfig &config);
	virtual ~SddpSolver(void);

	virtual void Solve(mat &weights, double &objective);
	void Solve(mat &weights, double &lower_bound_exact, double &upper_bound_mean, double &upper_bound_bound);
        void Solve(mat &weights, double &lower_bound_exact, double &upper_bound_mean, double &upper_bound_bound, vector<mat> &future_weights);
	virtual void GetStageSamples(vector<unsigned int> &stage_samples);
	virtual void GetReducedSamples(vector<unsigned int> &stage_samples);

        void EvaluatePolicy(boost::function<vector<vector<double> >
             (vector<const double *>)> policy, double &return_mean, double &return_variance, double &return_upper_bound, unsigned int iterations = 10);
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
	void AddCut(SddpSolverNode *node, unsigned int next_state);
	void ForwardPassStandard(unsigned int count, vector<SCENINDEX> &nodes, bool solve = true);
	void ForwardPassConditional(unsigned int count, vector<SCENINDEX> &nodes, bool solve = true);
	void ForwardPassFixed(unsigned int count, vector<SCENINDEX> &nodes, bool solve = true);
	void BackwardPass(const vector<SCENINDEX> &nodes);
	SddpSolverNode *BuildNode(TreeNode treenode);
	SddpSolverNode *BuildNode(SCENINDEX index);
	bool NodeExists(TreeNode treenode);
	bool NodeExists(SCENINDEX index);
	void CalculateSinglePathUpperBound(SddpSolverNode *node, double &value, double &probability);
	double ApproximateDecisionValue(SddpSolverNode *node);
	double ApproximateDecisionValue(SddpSolverNode *node, SddpSolverNode *parent);
	void ClearNodeDescendants(SddpSolverNode *node);
	void ClearSingleNode(SddpSolverNode *node);
	void ChooseRandomIndices(SCENINDEX start, SCENINDEX end, unsigned int count, vector<SCENINDEX> & indices);
	void ChooseRandomSubset(const vector<SCENINDEX> & set, unsigned int count, vector<SCENINDEX> & subset);
	void ConnectNode(SddpSolverNode *node, SddpSolverNode *parent);
	double CalculateUpperBoundDefault(const vector<SCENINDEX> &nodes);
	double CalculateUpperBoundConditional(const vector<SCENINDEX> &nodes);
        void CalculateFutureWeights(const vector<SCENINDEX> &nodes, vector<mat> &future_weights);
	double GetNodeProbability(SddpSolverNode *node);
	double GetConditionalProbability(SddpSolverNode *node);
	double GetConditionalProbability(unsigned int stage);
	SddpSolverNode *SampleNode(SddpSolverNode *parent, unsigned int next_state);
	unsigned int SampleState(SddpSolverNode *parent);

	map<SCENINDEX, SddpSolverNode *> nodes_;
	vector<vector<SddpSolverCut> > cuts_;
	RandomGenerator *generator_;
	unsigned int forward_iterations_; //iteration nr of the algorithm
	unsigned int backward_iterations_; //iteration nr of the algorithm

	unsigned int solved_lps_; //number of LPs solved
	time_duration lp_solve_time_; //time spent in a LP solver
	
	SddpSolverConfig config_; //algorithm configuration

	//output files
	boost::filesystem::ofstream value_file_;
	boost::filesystem::ofstream value_file_forward_;
	map<SCENINDEX, bool> reported_forward_values_;
};
