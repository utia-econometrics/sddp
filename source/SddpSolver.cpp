#include "SddpSolver.h"
#include "NormalDistribution.h"
#include "DiscreteDistribution.h"

#include <queue>

#define DEBUG_FORWARD 0
#define DEBUG_TREE 0
#define DEBUG_BOUND 0
#define DEBUG_CPLEX 0
#define DEBUG_BOUND_COUNT 100
#define DEBUG_BOUND_NODES 10000 //10000
#define STOP_CUTS 0
#define REPORT_COMPUTATION_TIMES 0
#define REPORT_NODE_VALUES 0
#define DEBUG_WRITE_MODEL 0

const double SddpSolver::CONVERGENCE_BOUND = 0.01;
#if DEBUG_TREE == 1
	const unsigned int SddpSolver::FORWARD_COUNT = 1;
	const unsigned int SddpSolver::BACKWARD_COUNT = 1;
#else
	const unsigned int SddpSolver::FORWARD_COUNT = 10000; //10000
	const unsigned int SddpSolver::BACKWARD_COUNT = 10; //25
#endif
const unsigned int SddpSolver::MAX_ITERATIONS = 200;
const double SddpSolver::CONDITIONAL_PROBABILITY = 0.3; //0.3

#ifdef _DEBUG
	const ExternalSolver SddpSolver::EXTERNAL_SOLVER = SOLVER_COINOR;
#else
	//const ExternalSolver SddpSolver::EXTERNAL_SOLVER = SOLVER_COINOR;
	const ExternalSolver SddpSolver::EXTERNAL_SOLVER = SOLVER_CPLEX;
#endif

const double SddpSolver::STOP_CONFIDENCE = 0.95;
const double SddpSolver::EPSILON = 1e-8; //1e-8;
const int SddpSolver::STOP_ELEMENTS = 10;

//important settings, set all to true for the best results
const bool SddpSolver::COMPARISON_EQUAL = true; //true
const bool SddpSolver::USE_IMPORTANCE = true; //true

//heuristic optimize the nodes selection in first stages .. implementation not ideally finished for importance/conditional
//needs to be turned off for inequal probabilities of nodes!!
const bool SddpSolver::QUICK_CONDITIONAL = false; //false

//conditional provides better results
const SddpStrategy SddpSolver::SOLVER_STRATEGY = STRATEGY_CONDITIONAL;
//const SddpStrategy SddpSolver::SOLVER_STRATEGY = STRATEGY_DEFAULT;

//noder per stage in the forward pass estimators
const int SddpSolver::FORWARD_FIXED_STAGE_NODES = 0;


SddpSolver::SddpSolver(const ScenarioModel *model, unsigned int fix_descendant_count, unsigned int fix_reduced_count)
:Solver(model)
{
	tree_ = 0;
	cuts_.resize(model_->GetStagesCount());
	generator_ = RandomGenerator::GetGenerator();

	expectation_coefficients_ = model_->GetParam().expectation_coefficients;
	risk_coefficients_ = model_->GetParam().risk_coefficients;
	risk_coefficients_other_ = model_->GetParam().risk_coefficients_other;
	fix_descendant_count_ = fix_descendant_count;
	fix_reduced_count_ = fix_reduced_count;

	if(BACKWARD_COUNT > FORWARD_COUNT) {
		throw SddpSolverException("Invalid configuration, there have to be more forward pass nodes than backward pass nodes.");
	}

	if(model_->GetParam().confidence > model_->GetParam().confidence_other) {
		throw SddpSolverException("Invalid configuration, second confidence level has to be higher than the first.");
	}
}

SddpSolver::~SddpSolver(void)
{
	DestroySolverTree();
	delete generator_;
}

void SddpSolver::DestroySolverTree() {
	ClearNodes();
}

void SddpSolver::ClearNodeDescendants(SddpSolverNode *node) {
	for(unsigned int i = 0; i < node->descendants.size(); ++i) {
		ClearNodeDescendants(node->descendants[i]); //allows to clear all at once
		nodes_.erase(node->descendants[i]->tree_node.GetNumber());
		delete node->descendants[i];
	}
	node->descendants.clear();
}

void SddpSolver::ClearSingleNode(SddpSolverNode *node) {
#ifdef _DEBUG
	if(node->descendants.size() > 0) {
		throw SddpSolverException("Cannot delete node with descendants.");
	}
#endif
	nodes_.erase(node->tree_node.GetNumber());
	delete node;
}

void SddpSolver::ClearNodes() {
	map<SCENINDEX, SddpSolverNode *>::iterator it;
	for(it = nodes_.begin(); it != nodes_.end(); ++it) {
		delete it->second;
	}
	nodes_.clear();
}

void SddpSolver::BuildSolverTree() {
	//build the scenario tree
	GetTree();
}

void SddpSolver::Solve(mat &weights, double &objective) {
	double dummy1;
	double dummy2;
	Solve(weights, objective, dummy1, dummy2);
}

void SddpSolver::Solve(arma::mat &weights, double &lower_bound_exact, double &upper_bound_mean, double &upper_bound_bound) {
	//scenario tree is needed
	BuildSolverTree();

	forward_iterations_ = 0;
	backward_iterations_ = 0;
	solved_lps_ = 0;
	lp_solve_time_ = milliseconds(0);

	vector<SCENINDEX> forward_nodes;
	vector<SCENINDEX> backward_nodes;

	vector<double> lower_bounds;
	vector<double> upper_bounds;
	vector<long> computation_times_backward_pass;
	vector<long> computation_times_forward_pass;

	SddpSolverNode * node;
	double upper_bound;
	double lower_bound;
	double ub_margin;
	double ub_mean;

	bool do_backward_pass = true;

	//vars to choose nodes
	unsigned int laststage = model_->GetStagesCount();
	SCENINDEX from = tree_->ScenarioCount(laststage - 1);
	SCENINDEX to = from + tree_->ScenarioCountStage(laststage) - 1;

	while(forward_iterations_ < MAX_ITERATIONS) {
#if REPORT_NODE_VALUES == 1
		value_file_.open("value.txt", ios_base::out);
#endif

		//go forward and calculate the price
		++forward_iterations_;
		ptime fw_start = microsec_clock::local_time();
		unsigned int fw_start_lps = solved_lps_;
		time_duration fw_start_lp_times = lp_solve_time_;
		cout << "Running forward iteration " << forward_iterations_ << ".. ";

		unsigned int fw_count = FORWARD_COUNT;
#if DEBUG_BOUND == 1
		if(!do_backward_pass) {
			fw_count = DEBUG_BOUND_NODES;
		}
#endif
		if(SOLVER_STRATEGY == STRATEGY_DEFAULT) {
			//choose the nodes
			ForwardPassStandard(fw_count, forward_nodes);
			upper_bound = CalculateUpperBoundDefault(forward_nodes);
		}
		else if(SOLVER_STRATEGY == STRATEGY_CONDITIONAL) {
			ForwardPassConditional(fw_count, forward_nodes);
			upper_bound = CalculateUpperBoundConditional(forward_nodes);
		}
		else {
			throw SddpSolverException("Strategy undefined.");
		}

		upper_bounds.push_back(upper_bound);
		time_duration fw_elapsed = microsec_clock::local_time() - fw_start;
		computation_times_forward_pass.push_back(fw_elapsed.total_milliseconds());
		unsigned int fw_solved_lps = solved_lps_ - fw_start_lps;
		time_duration fw_lp_times = lp_solve_time_ - fw_start_lp_times;
		cout << "completed with upper bound " << upper_bound << " @ " << fw_elapsed.total_milliseconds() << " ms" << endl;
		cout << "LP stats - count: " << fw_solved_lps << " @ " << fw_lp_times.total_milliseconds() << " ms" << endl;


#if DEBUG_BOUND == 1
		if(!do_backward_pass && (upper_bounds.size() - lower_bounds.size() >= DEBUG_BOUND_COUNT)) {
			cout << "Iteration limit reached, stopping." << endl;
			break;
		}
#endif

		//go backward, get the cuts and lower bound
		if(do_backward_pass) {
			++backward_iterations_;
			ptime bw_start = microsec_clock::local_time();
			unsigned int bw_start_lps = solved_lps_;
			time_duration bw_start_lp_times = lp_solve_time_;
			cout << "Running backward iteration " << backward_iterations_ << ".. ";
			//select backward count of nodes
			ChooseRandomSubset(forward_nodes, BACKWARD_COUNT, backward_nodes);
			BackwardPass(backward_nodes);

			node = BuildNode(0);
			SolveNode(node);
			lower_bound = node->objective;
			lower_bounds.push_back(node->objective);
			time_duration bw_elapsed = microsec_clock::local_time() - bw_start;
			computation_times_backward_pass.push_back(bw_elapsed.total_milliseconds());
			unsigned int bw_solved_lps = solved_lps_ - bw_start_lps;
			time_duration bw_lp_times = lp_solve_time_ - bw_start_lp_times;
			cout << "completed with lower bound " << lower_bound << " @ " << bw_elapsed.total_milliseconds() << " ms" << endl;
			cout << "LP stats - count: " << bw_solved_lps << " @ " << bw_lp_times.total_milliseconds() << " ms" << endl;

			//stopping rules

			//needs to go first in order to calculate ub stats
			if(upper_bounds.size() >= STOP_ELEMENTS) {
				double variance = 0.0;
				ub_mean = 0.0;
				int elements = 0;
				unsigned int index = upper_bounds.size();
				while(elements < STOP_ELEMENTS) {
					++elements;
					--index;
					ub_mean += upper_bounds[index];
					if(index == 0) {
						break;
					}
				}
				ub_mean /= elements;
				index = upper_bounds.size();
				int counter = 0;
				while(counter < elements) {
					++counter;
					--index;
					variance += pow(upper_bounds[index] - ub_mean, 2.0);
				}
				variance /= elements;
				NormalDistribution nd(zeros(1), ones(1, 1));
				ub_margin = ub_mean + sqrt(variance) * nd.InverseDistributionFunction(STOP_CONFIDENCE);
				cout << "Stopping margin: " << ub_margin << endl;
#if DEBUG_BOUND == 0
				if(ub_margin - lower_bound < CONVERGENCE_BOUND) {
					cout << "Upper bound and lower bound difference lower than required precision, stopping." << endl;
					break;
				}
#endif
			}

			//need to check upper bounds size also, because of the UB mean calculation
			//add one to the size for stabilization of the upper bound in the case of population mean
			if ((lower_bounds.size() >= STOP_ELEMENTS + 1) && (upper_bounds.size() >= STOP_ELEMENTS + 1) && (lower_bounds[lower_bounds.size() - 1] - lower_bounds[lower_bounds.size() - STOP_ELEMENTS - 1] < EPSILON)) {
#if DEBUG_BOUND == 1
	#if STOP_CUTS == 0 
				do_backward_pass = false;
	#endif
#else
				cout << "Lower bound not improving, stopping." << endl;
				break;
#endif
			}

#if STOP_CUTS > 0 
			//no cuts in the last stage, select the ones from the stage before that
			if(cuts_[model_->GetStagesCount() - 2].size() >= STOP_CUTS) {
				do_backward_pass = false;
			}
#endif
		}

		//clean up is needed at least where no backward pass is done
		//nodes would remain as descendants and would be counted to upper bound estimators
		ClearNodeDescendants(BuildNode(0));

#if REPORT_NODE_VALUES == 1
		value_file_.close();
#endif
		
	}

#if REPORT_NODE_VALUES == 1
	value_file_.close();
#endif

	if(forward_iterations_ == MAX_ITERATIONS) {
		cout << "Iteration limit reached, stopping." << endl;
	}

	cout << endl;
	unsigned int index = 0;
	for(; index < lower_bounds.size(); ++index) {
		cout << lower_bounds[index] << " < " << upper_bounds[index] << endl;
	}
	cout << endl;
	for(; index < upper_bounds.size(); ++index) {
		cout << " < " << upper_bounds[index] << endl;
	}
	cout << endl;

#if REPORT_COMPUTATION_TIMES == 1
	cout << endl;
	cout << "Times:" << endl;
	index = 0;
	for(; index < computation_times_backward_pass.size(); ++index) {
		cout << computation_times_backward_pass[index] << " < " << computation_times_forward_pass[index] << endl;
	}
	cout << endl;
	for(; index < computation_times_forward_pass.size(); ++index) {
		cout << " < " << computation_times_forward_pass[index] << endl;
	}
	cout << endl;
#endif

	node = BuildNode(0);
	SolveNode(node);
	weights.reshape(1, model_->GetAssetsCount());
	for(unsigned int i = 0; i < model_->GetAssetsCount(); ++i) {
		weights(i) = node->solution[i];
	}
	lower_bound_exact = node->objective;
	upper_bound_mean = ub_mean;
	upper_bound_bound = ub_margin;

#if DEBUG_BOUND == 1
	double sum = 0;
	unsigned int i = 1;
	for( ; i <= DEBUG_BOUND_COUNT; ++i) {
		if(i > upper_bounds.size()) {
			break;
		}
		sum += upper_bounds[upper_bounds.size() - i];
	}
	unsigned int count = i - 1;
	double mean = sum / count;

	sum = 0;
	for( ; i > 0; --i) {
		sum += pow(upper_bounds[upper_bounds.size() - i] - mean, 2);
	}
	double sd = sqrt(sum / (count - 1));
	cout << "Upper bound stats: " << mean << " (" << sd << ")" << endl;

	sum = 0;
	i = 1;
	for( ; i <= DEBUG_BOUND_COUNT; ++i) {
		if(i > computation_times_forward_pass.size()) {
			break;
		}
		sum += computation_times_forward_pass[computation_times_forward_pass.size() - i];
	}
	count = i - 1;
	mean = sum / count;

	sum = 0;
	for( ; i > 0; --i) {
		sum += pow(computation_times_forward_pass[computation_times_forward_pass.size() - i] - mean, 2);
	}
	sd = sqrt(sum / (count - 1));
	cout << "Computation time stats [ms]: " << mean << " (" << sd << ")" << endl;
#endif
}

void SddpSolver::EvaluatePolicy(boost::function<vector<vector<double>>(vector<const double *>)> policy, double &return_mean, double &return_upper_bound, unsigned int iterations) {
	vector<double> upper_bounds;
	double upper_bound;
	vector<SCENINDEX> forward_nodes;
	unsigned int fw_count = FORWARD_COUNT;
#if DEBUG_BOUND == 1
	fw_count = DEBUG_BOUND_NODES;
#endif
	for(unsigned int i = 0; i < iterations; ++i){
		//choose the nodes
		if(SOLVER_STRATEGY == STRATEGY_DEFAULT) {
			ForwardPassStandard(fw_count, forward_nodes);
		}
		else if(SOLVER_STRATEGY == STRATEGY_CONDITIONAL) {
			ForwardPassConditional(fw_count, forward_nodes);
		}
		else {
			throw SddpSolverException("Strategy undefined.");
		}

		//get solutions
		for(unsigned int i = 0; i < forward_nodes.size(); ++i) {
			SddpSolverNode *base_node = BuildNode(forward_nodes[i]);
			vector<const double *> scenario;
			scenario.resize(model_->GetStagesCount());
			unsigned int index = model_->GetStagesCount() - 1;
			SddpSolverNode *node = base_node;
			while(node != 0) {
				TreeNode tree_node = node->tree_node;
				const double *values = tree_node.GetValues();
				scenario[index] = values;
				--index;
				node = node->parent;
			}
			vector<vector<double>> solutions = policy(scenario);
			node = base_node;
			index = model_->GetStagesCount() - 1;
			while(node != 0) {
				//full copy of the weights
				unsigned int j = 0;
				for(; j < model_->GetAssetsCount(); ++j) {
					node->solution[j] = solutions[index][j];
				}
				//var level
				node->var = solutions[index][j];
				--index;
				node = node->parent;
			}
		}

		//evaluate nodes
		if(SOLVER_STRATEGY == STRATEGY_DEFAULT) {
			//choose the nodes
			upper_bound = CalculateUpperBoundDefault(forward_nodes);
		}
		else if(SOLVER_STRATEGY == STRATEGY_CONDITIONAL) {
			upper_bound = CalculateUpperBoundConditional(forward_nodes);
		}
		else {
			throw SddpSolverException("Strategy undefined.");
		}

		upper_bounds.push_back(upper_bound);
	}

	//stats and return
	double variance = 0.0;
	double mean = 0.0;
	for(unsigned int i = 0; i < iterations; ++i) {
		mean += upper_bounds[i];
	}
	mean /= iterations;
	for(unsigned int i = 0; i < iterations; ++i) {
		variance += pow(upper_bounds[i] - mean, 2.0);
	}
	variance /= iterations;
	NormalDistribution nd(zeros(1), ones(1, 1));
	double margin = mean + sqrt(variance) * nd.InverseDistributionFunction(STOP_CONFIDENCE);

	return_mean = mean;
	return_upper_bound = margin;
}

vector<vector<double>> SddpSolver::GetPolicy(vector<const double *> scenario) {
	//build a single path tree
	unsigned int assets = model_->GetAssetsCount();
	unsigned int stages = model_->GetStagesCount();
	vector<Distribution *> distributions;
	for(unsigned int stage = 1; stage <= stages; ++stage) {
		mat stage_scen(scenario[stage - 1], 1, assets);
		distributions.push_back(new DiscreteDistribution(stage_scen));
	}
	vector<unsigned int> stage_samples;
	for(unsigned int stage = 1; stage <= stages; ++stage) {
		stage_samples.push_back(1); //fixed one exact scenario
	}
	ScenarioTree *tree = new ScenarioTree(stages, stage_samples, assets, STAGE_INDEPENDENT, distributions, 0);
	tree->GenerateTree();

	//clean the distributions
	for(unsigned int i = 0; i < distributions.size(); ++i) {
		delete distributions[i];
	}

	//build a solver node structures
	vector<SddpSolverNode *> nodes;
	for(unsigned int index = 0; index < stages; ++index) {
		//TODO: ugly code copy
		SddpSolverNode *node = new SddpSolverNode();
		TreeNode treenode = (*tree)(index);
		node->tree_node = treenode;
		node->solution = new double[assets];
		node->subgradient = new double[assets];
		nodes.push_back(node);
		if(index > 0) {
			ConnectNode(node, nodes[index - 1]);
		}
	}

	//solve
	for(unsigned int i = 0; i < nodes.size(); ++i) {
		SolveNode(nodes[i]);
	}

	//return
	vector<vector<double>> solutions;
	for(unsigned int i = 0; i < nodes.size(); ++i) {
		vector<double> stage_soln;
		for(unsigned int a = 0; a < assets; ++a) {
			stage_soln.push_back(nodes[i]->solution[a]);
		}
		stage_soln.push_back(nodes[i]->var); //var level to solns
		solutions.push_back(stage_soln);
	}

	//clean up & return
	delete tree;
	for(int i = nodes.size() - 1; i >= 0; --i) {
		delete nodes[i];
	}
	
	return solutions;
}

double SddpSolver::CalculateUpperBoundDefault(const vector<SCENINDEX> &nodes) {
	vector<SddpSolverNode *> parent_nodes;
	vector<SddpSolverNode *> actual_nodes;

	if (model_->GetParam().risk_measure != RISK_CVAR_NESTED) {
		throw SddpSolverException("Not implemented yet");
	}
#if REPORT_NODE_VALUES == 1
	value_file_forward_.open("value_forward.txt", ios_base::out);
	reported_forward_values_.clear();
#endif
	
	//fill the nodes
	for(unsigned int i = 0; i < nodes.size(); ++i) {
		SddpSolverNode *node = BuildNode(nodes[i]);
#ifdef _DEBUG
		if(node->parent == 0) {
			throw SddpSolverException("Cannot calculate upper bound with no parent nodes.");
		}
#endif
		if(find(actual_nodes.begin(), actual_nodes.end(), node->parent) == actual_nodes.end()) {
			actual_nodes.push_back(node->parent);
		}
		node->value = -node->GetValue();
	}
	
	vector<double>::reverse_iterator exp_c = expectation_coefficients_.rbegin();
	vector<double>::reverse_iterator risk_c = risk_coefficients_.rbegin();
	vector<double>::reverse_iterator risk_c_other = risk_coefficients_other_.rbegin();
	double conf_inv = 1 - model_->GetParam().confidence;
	double conf_inv_other = 1 - model_->GetParam().confidence_other;

	while(actual_nodes.size() > 0) {
		parent_nodes.clear();
		for(unsigned int i = 0; i < actual_nodes.size(); ++i) {
			
			SddpSolverNode *node = actual_nodes[i];

#if REPORT_NODE_VALUES == 1
			//if(reported_forward_values_.find(node->tree_node.GetIndex()) == reported_forward_values_.end()) {
				value_file_forward_ << node->tree_node.GetStage() <<  "\t" << node->value << endl;
			//	reported_forward_values_[node->tree_node.GetIndex()] = true;
			//}
#endif

			double value = 0.0;
			double recourse = 0.0;
			double recourse_other = 0.0;
			for(unsigned int j = 0; j < node->descendants.size(); ++j) {
				SddpSolverNode *descendant = node->descendants[j];
				if(descendant->value > node->var) {
					recourse += (descendant->value - node->var);
				}
				if(descendant->value > node->var_other) {
					recourse_other += (descendant->value - node->var_other);
				}
				value += descendant->value;
			}
			value *= *exp_c;
			recourse *= *risk_c / conf_inv;
			recourse_other *= *risk_c_other / conf_inv_other;
			value += recourse;
			value += recourse_other;
			value /= node->descendants.size();
			
			value += *risk_c * node->var;
			value += *risk_c_other * node->var_other;

			if(node->parent != 0) {
				value += -node->GetValue();
				if(find(parent_nodes.begin(), parent_nodes.end(), node->parent) == parent_nodes.end()) {
					actual_nodes.push_back(node->parent);
				}
			}
			node->value = value;	
		}
		++risk_c;
		++risk_c_other;
		++exp_c;
		actual_nodes = parent_nodes;
	}

#if REPORT_NODE_VALUES == 1
	value_file_forward_.close();
#endif


	return BuildNode(0)->value;

}

double SddpSolver::CalculateUpperBoundConditional(const vector<SCENINDEX> &nodes) {
	double upper_bound = 0.0;
	double total_prop = 0.0;
#if REPORT_NODE_VALUES == 1
	value_file_forward_.open("value_forward.txt", ios_base::out);
	reported_forward_values_.clear();
#endif
	for(unsigned int i = 0; i < nodes.size(); ++i) {
		SddpSolverNode *node = BuildNode(nodes[i]);
		double val;
		double prop;
		CalculateTotalValue(node, val, prop);
		upper_bound += prop * val;
		total_prop += prop;
	}
#if REPORT_NODE_VALUES == 1
	value_file_forward_.close();
#endif
	//total_prop adds stability
	upper_bound /= total_prop;
	return upper_bound;
}

bool SddpSolver::NodeExists(SCENINDEX index) {
	TreeNode tnode = (*tree_)(index);
	return NodeExists(tnode);
}

bool SddpSolver::NodeExists(TreeNode treenode) {
	return nodes_.find(treenode.GetNumber()) != nodes_.end();
}

SddpSolverNode *SddpSolver::BuildNode(SCENINDEX index) {
	TreeNode tnode = (*tree_)(index);
	return BuildNode(tnode);
}

SddpSolverNode *SddpSolver::BuildNode(TreeNode treenode) {
	if(NodeExists(treenode)) {
		return nodes_[treenode.GetNumber()];
	}

	SddpSolverNode *node = new SddpSolverNode();
	unsigned int assets = model_->GetAssetsCount();
	node->tree_node = treenode;
	node->solution = new double[assets];
	node->subgradient = new double[assets];
	nodes_[treenode.GetNumber()] = node;
	return node;
}

void SddpSolver::ChooseRandomIndices(SCENINDEX start, SCENINDEX end, unsigned int count, vector<SCENINDEX> & indices) {
	if(end - start + 1 < count) {
		throw SddpSolverException("Cannot choose more numbers than there are in given interval.");
	}
	indices.clear();
	map<SCENINDEX, bool> chosen;
	for(unsigned int i = 0; i < count; ++i) {
		SCENINDEX index = generator_->GetRandomHuge(start, end);
		if(chosen.find(index) == chosen.end()) {
			indices.push_back(index);
			chosen[index] = true;
		}
		else {
			--i; //reiterate
		}
	}
}

void SddpSolver::ChooseRandomSubset(const vector<SCENINDEX> & set, unsigned int count, vector<SCENINDEX> & subset) {
	if(set.size() < count) {
		throw SddpSolverException("Cannot choose more nods than given.");
	}
	subset.clear();
	map<SCENINDEX, bool> chosen;
	unsigned int size = set.size();
	for(unsigned int i = 0; i < count; ++i) {
		unsigned int index = generator_->GetRandomInt(0, size - 1);
		if(chosen.find(index) == chosen.end()) {
			subset.push_back(set[index]);
			chosen[index] = true;
		}
		else {
			--i; //reiterate
		}
	}
}

void SddpSolver::ConnectNode(SddpSolverNode *node, SddpSolverNode *parent) {
	node->parent = parent;
	if(find(parent->descendants.begin(), parent->descendants.end(), node) == parent->descendants.end()) {
		//on upper parts of tree we could have hit the path we were already on
		//TODO: speed up?
		parent->descendants.push_back(node);
	}
}

void SddpSolver::ForwardPassStandard(unsigned int count, vector<SCENINDEX> &nodes, bool solve) {	
	nodes.clear();
	unsigned int stages = model_->GetStagesCount();
	vector<SddpSolverNode *> actual_nodes;
	vector<SddpSolverNode *> last_nodes;

	//root
	SddpSolverNode *parent = BuildNode(0);
	SolveNode(parent);
	last_nodes.push_back(parent);

	//approximate the nodes count for each stage
	unsigned int stage_count = static_cast<unsigned int>(pow(static_cast<double>(count), 1.0/(stages - 1))) + 1;
	if(FORWARD_FIXED_STAGE_NODES > 0 && stage_count < FORWARD_FIXED_STAGE_NODES) {
		stage_count = FORWARD_FIXED_STAGE_NODES;
	}

	for(unsigned int stage = 2; stage <= stages; ++stage) {
		actual_nodes.clear();
#if DEBUG_FORWARD == 1
		if(true) {
#elif DEBUG_TREE == 1
		if(true) {
#else
		if(QUICK_CONDITIONAL && (tree_->DescendantCountStage(stage) <= stage_count)) {
#endif
			//choose all nodes
			SCENINDEX from = tree_->ScenarioCount(stage - 1);
			SCENINDEX to = from + tree_->ScenarioCountStage(stage) - 1;
			for(SCENINDEX index = from; index <= to; ++index) {
				SddpSolverNode *node = BuildNode(index);
				ConnectNode(node, BuildNode(node->tree_node.GetParent().GetIndex()));
				if(solve) {
					SolveNode(node);
				}
				actual_nodes.push_back(node);
			}
		}
		else {
			for(unsigned int i = 0; i < last_nodes.size(); ++i) {
				SddpSolverNode *parent = last_nodes[i];
				for(unsigned int j = 0; j < stage_count; ++j) {
					SddpSolverNode *node = SampleNodeByProbability(parent, generator_->GetRandom());
					ConnectNode(node, parent);
					if(solve) {
						SolveNode(node);
					}
					actual_nodes.push_back(node);
				}
			}
		}
		last_nodes = actual_nodes;
	}

	for(unsigned int i = 0; i < last_nodes.size(); ++i) {
		nodes.push_back(last_nodes[i]->tree_node.GetIndex());
	}
}

SddpSolverNode * SddpSolver::SampleNodeByProbability(SddpSolverNode *parent, double probability) {
	double prop_sum = 0;
	ScenarioTreeConstIterator it;
	SddpSolverNode *node;
	for(it = parent->tree_node.GetDescendantsBegin(); it != parent->tree_node.GetDescendantsEnd(); ++it) {
		prop_sum += it->GetProbability();
		if(prop_sum >= probability) {
			node = BuildNode(*it);
			break;
		}
	}
	return node;
}

void SddpSolver::ForwardPassConditional(unsigned int count, vector<SCENINDEX> &nodes, bool solve) {	
	nodes.clear();
	unsigned int stages = model_->GetStagesCount();
	vector<SddpSolverNode *> actual_nodes;
	vector<SddpSolverNode *> last_nodes;

	//approximate the stage count
	unsigned int stage_count = (count / (stages - 1)) + 1;
	if(FORWARD_FIXED_STAGE_NODES > 0 && stage_count < FORWARD_FIXED_STAGE_NODES) {
		stage_count = FORWARD_FIXED_STAGE_NODES;
	}

	//root
	SddpSolverNode *parent = BuildNode(0);
	SolveNode(parent);
	last_nodes.push_back(parent);

	for(unsigned int stage = 2; stage <= stages; ++stage) {
		actual_nodes.clear();
#if DEBUG_FORWARD == 1
		if(true) {
#elif DEBUG_TREE == 1
		if(true) {
#else
		if(QUICK_CONDITIONAL && (tree_->ScenarioCountStage(stage) <= stage_count)) {
#endif
			//choose all
			SCENINDEX from = tree_->ScenarioCount(stage - 1);
			SCENINDEX to = from + tree_->ScenarioCountStage(stage) - 1;
			for(SCENINDEX index = from; index <= to; ++index) {
				SddpSolverNode *node = BuildNode(index);
				ConnectNode(node, BuildNode(node->tree_node.GetParent().GetIndex()));
				if(solve) {
					SolveNode(node);
				}
				actual_nodes.push_back(node);
			}
		}
		else {
			//conditional selection
			unsigned int counter = 0;
			unsigned int parent_idx = 0;
			vector<SddpComparableNode> nodes;
			SddpSolverNode *parent = 0;
			SddpSolverNode *node;
			double cut_margin;
			unsigned int margin_index;
			double margin_probability;
			while(counter < stage_count) {
				if(USE_IMPORTANCE) {
					if(parent != last_nodes[parent_idx]) { //optimization when we select multiple nodes for a single parent
						parent = last_nodes[parent_idx];
						nodes.clear();
						ScenarioTreeConstIterator it;
						for(it = parent->tree_node.GetDescendantsBegin(); it != parent->tree_node.GetDescendantsEnd(); ++it) {
							bool existed = NodeExists(*it);
							node = BuildNode(*it);
							//memory optimization - do not connect, calculate value directly using parent and delete
							double capital = CalculateNodeCapital(node, parent);
							nodes.push_back(SddpComparableNode(node->tree_node.GetIndex(), capital, node->GetProbability()));
							if(!existed) {
								//only if it was not selected before we can delete it
								ClearSingleNode(node);
							}
						}
						sort(nodes.begin(), nodes.end());

						//marginal node to select
						double prop_sum = 0;
						for(unsigned int i = 0; i < nodes.size(); ++i) {
							prop_sum += nodes[i].probability;
							if(prop_sum >= model_->GetParam().confidence) {
								margin_index = i;
								margin_probability = prop_sum - nodes[i].probability; //will be incremented by margin index
								cut_margin = nodes[i].value;
								cut_margin *= 1 + model_->GetParam().transaction_costs; //sell all
								cut_margin /= 1 - model_->GetParam().transaction_costs; //buy something else
								break;
							}
						}
					}
					//select the node
					if(generator_->GetRandom() < GetConditionalProbability(stage)) { //coin flip
						//conditional node
						double prop_gen = generator_->GetRandom() * (1 - margin_probability) + margin_probability;
						double prop_sum = margin_probability;
						for(unsigned int i = margin_index; i < nodes.size(); ++i) {
							prop_sum += nodes[i].probability;
							if(prop_sum >= prop_gen) {
								node = BuildNode(nodes[i].index);
								node->type = NODE_CVAR;
								break;
							}
						}
					}
					else {
						unsigned int selected;
						double prop_gen = generator_->GetRandom() * margin_probability;
						double prop_sum = 0;
						for(unsigned int i = 0; i <= margin_index; ++i) {
							prop_sum += nodes[i].probability;
							if(prop_sum >= prop_gen) {
								selected = i;
								node = BuildNode(nodes[i].index);
								node->type = NODE_EXPECTATION;
								break;
							}
						}
						
						//margin according to the trans costs
						if(COMPARISON_EQUAL) {
							if(nodes[selected].value < cut_margin) {
								//we can cut off the downside risk when calculating upper bound
								node->cut_tail = true;
							}
						}
					}
				}
				else {
					//not using the importance sampling density
					parent = last_nodes[parent_idx];
					node = SampleNodeByProbability(parent, generator_->GetRandom());	
				}
				//solve the selected node and continue
				ConnectNode(node, parent);
				if(solve) {
					SolveNode(node);
				}
				actual_nodes.push_back(node);
				++parent_idx;
				if(parent_idx >= last_nodes.size()) {
					//reiterate parents
					parent_idx = 0;
				}
				++counter;
			}
		}
		last_nodes = actual_nodes;
	}

	for(unsigned int i = 0; i < last_nodes.size(); ++i) {
		nodes.push_back(last_nodes[i]->tree_node.GetIndex());
	}
}

void SddpSolver::BackwardPass(const vector<SCENINDEX> &nodes) {
	//solve the in the order of the stages to get the best cuts
	//the nodes passed should be from the last stage
	vector<SCENINDEX> solvenodes;
	vector<SCENINDEX> actualnodes;
	actualnodes = nodes;

	while(!actualnodes.empty()) {
		solvenodes.clear();
		for(unsigned int i = 0; i < actualnodes.size(); ++i) {
			SddpSolverNode *lastnode = BuildNode(actualnodes[i]);
			if(lastnode->tree_node.GetDescendantsCount() > 0) {
				//process the cuts
				AddCut(lastnode);
			}
			if(lastnode->parent != 0) {
				//will be solved in next round
				solvenodes.push_back(lastnode->parent->tree_node.GetIndex());
			}
		}
		actualnodes = solvenodes;
	}
}

void SddpSolver::SolveNodeCplex(SddpSolverNode *node) {
#ifndef _DEBUG
	//assets count N
	unsigned int assets = model_->GetAssetsCount();
	unsigned int stage = node->tree_node.GetStage();
	bool root_node = (stage == 1);
	bool last_stage = (stage == model_->GetStagesCount());
	double conf_inv = 1 - model_->GetParam().confidence;
	double conf_inv_other = 1 - model_->GetParam().confidence_other;

	IloEnv env;
	IloModel model(env);
	IloCplex cplex(model);
	IloExpr obj_expr(env);

	//no output
	cplex.setOut(env.getNullStream());

	//var x1, ..., xN = weights 1 .. N
	IloNumVarArray x(env, assets);
	for(unsigned int i = 0; i < assets; ++i) {
		//zero lower bound
		x[i] = IloNumVar(env, 0.0);
		
		//max profit = min negative loss
		if(!root_node) {
			switch(model_->GetParam().risk_measure) {
				case RISK_CVAR_NESTED:
					obj_expr += -1 * x[i];
					break;
				case RISK_CVAR_MULTIPERIOD:
					obj_expr += -1 * expectation_coefficients_[stage - 1] * x[i];
					break;
			}
		}
	}
	model.add(x);

	//var q = lower estimate for the cut
	IloNumVar q(env, GetRecourseLowerBound(stage), GetRecourseUpperBound(stage));
	if(!last_stage) {
		//last stage has no recourse
		obj_expr += model_->GetParam().discount_factor * q;
	}
	model.add(q);

	//var c = positive value in the cvar
	IloNumVar c(env, 0.0);
	if(!root_node && (model_->GetParam().risk_measure == RISK_CVAR_MULTIPERIOD)) {
		//objective with risk aversion coef lambda
		obj_expr += risk_coefficients_[stage-1] / conf_inv * c;
		model.add(c);
	}

	//var c_other = positive value in the cvar
	IloNumVar c_other(env, 0.0);
	if(!root_node && (model_->GetParam().risk_measure == RISK_CVAR_MULTIPERIOD)) {
		//objective with risk aversion coef lambda
		obj_expr += risk_coefficients_other_[stage-1] / conf_inv_other * c_other;
		model.add(c_other);
	}

	//var var = variable to calculate CVaR = VaR level
	IloNumVar var(env, GetRecourseLowerBound(stage), GetRecourseUpperBound(stage));
	//objective according to the risk aversion lambda
	if(!last_stage) {
		//last stage has no recourse
		obj_expr += risk_coefficients_[stage] * var;
	}
	model.add(var);

	//var var_other = variable to calculate CVaR = VaR level
	IloNumVar var_other(env, GetRecourseLowerBound(stage), GetRecourseUpperBound(stage));
	//objective according to the risk aversion lambda
	if(!last_stage) {
		//last stage has no recourse
		obj_expr += risk_coefficients_other_[stage] * var_other;
	}
	model.add(var_other);

	IloObjective obj = IloMinimize(env, obj_expr);
	model.add(obj);

	//transaction costs dummy variable
	IloNumVarArray trans(env, assets);
	for(unsigned int i = 0; i < assets; ++i) {
		//zero lower bound
		trans[i] = IloNumVar(env, 0.0);
	}
	if(!root_node) {
		model.add(trans);
	}

	//constraint capital = sum of the weights equals initial capital one
	IloExpr cap_expr(env);
	double transaction_coef = model_->GetParam().transaction_costs;
	for(unsigned int i = 0; i < assets; ++i) {
		cap_expr += x[i];
		if(!root_node) {
			cap_expr += transaction_coef * trans[i];
		}
	}
	double total_capital;
	if(root_node) {
		//no parent, init the capital with 1
		total_capital = 1.0;
	}
	else {
		//sum up the capital under this scenario
		total_capital = node->GetCapital(node->parent->solution);
	}
	IloRange capital(env, cap_expr, total_capital);
	model.add(capital);

	//contraint the positive part "c" of the CVaR value
	IloExpr cvar_expr(env);
	IloRange cvar;
	if(!root_node && (model_->GetParam().risk_measure == RISK_CVAR_MULTIPERIOD)) {
		cvar_expr -= c; //the varible itself
		for(unsigned int i = 0; i < assets; ++i) {
			cvar_expr -= x[i]; //-c transp x
		}
		cvar = IloRange(env, cvar_expr, node->parent->var); //previous value at risk
		model.add(cvar);
	}

	//contraint the positive part "c_other" of the CVaR value
	IloExpr cvar_expr_other(env);
	IloRange cvar_other;
	if(!root_node && (model_->GetParam().risk_measure == RISK_CVAR_MULTIPERIOD)) {
		cvar_expr_other -= c_other; //the varible itself
		for(unsigned int i = 0; i < assets; ++i) {
			cvar_expr_other -= x[i]; //-c transp x
		}
		cvar_other = IloRange(env, cvar_expr_other, node->parent->var_other); //previous value at risk
		model.add(cvar_other);
	}


	//constraint transaction costs
	IloRangeArray costs_pos(env);	
	IloRangeArray costs_neg(env);
	if(!root_node) {
		for(unsigned int i = 0; i < assets; ++i) {
			//current position in asset i
			double parent_value = node->tree_node.GetValues()[i] * node->parent->solution[i];

			//positive and negative part of linearization for absolute value
			IloExpr cost_expr_pos(env);
			IloExpr cost_expr_neg(env);
			cost_expr_pos += x[i];
			cost_expr_pos += -1 * trans[i];
			cost_expr_neg += -1 * x[i];
			cost_expr_neg += -1 * trans[i];
			costs_pos.add(IloRange(env, cost_expr_pos, parent_value));
			costs_neg.add(IloRange(env, cost_expr_neg, -parent_value));
		}
		model.add(costs_pos);
		model.add(costs_neg);
	}

	//add cuts
	IloRangeArray cuts(env);
	for(unsigned int i = 0; i < cuts_[stage - 1].size(); ++i) {
		SddpSolverCut cut = cuts_[stage - 1][i];
		IloExpr cut_expr(env);
		cut_expr += -1 * q;
		for(unsigned int j = 0; j < assets; ++j) {
			cut_expr += cut.gradient[j] * x[j];
		}
		cut_expr += cut.gradient[assets] * var;
		cut_expr += cut.gradient[assets + 1] * var_other;
		cuts.add(IloRange(env, cut_expr, -cut.absolute));
	}
	model.add(cuts);

#if DEBUG_CPLEX == 1
	if(node->parent == 0) {
		stringstream str;
		str << "FW_" << node->solved_forward_nr << "_BW_" << node->solved_forward_nr << ".lp";
		cplex.exportModel(str.str().c_str());
	}
#endif

	//performance tuning
	cplex.setParam(IloCplex::PreDual, 1);
	cplex.setParam(IloCplex::RootAlg, IloCplex::Primal);

	//solve
	if(!cplex.solve()) {
		cout << "No solution found!" << endl;
		throw SddpSolverException("Solution not found");
	}

	//solutions	
	for(unsigned int i = 0; i < assets; ++i) {
		node->solution[i] = cplex.getValue(x[i]);
	}

	//fill also the VaR value
	node->var = cplex.getValue(var); 

	//fill also the other VaR value
	node->var_other = cplex.getValue(var_other); 

	//fill the objective
	node->objective = cplex.getObjValue();

#if DEBUG_CPLEX == 1
	if(node->parent == 0) {
		cout << "Cplex objective: " << cplex.getObjValue() << endl;
	}
#endif

	//calculate subgradient
	//Shapiro SPBook Proposition 2.2 .. T = (-price1, ... -price N) ; W = (1, ..., 1), h = 0
	double dual_cap = cplex.getDual(capital);
	for(unsigned int i = 0; i < assets; ++i) {
		double subgrad = 0.0;
		double value = node->tree_node.GetValues()[i];

		//the capital part
		subgrad += dual_cap * value;
		
		//trans costs
		if(!root_node) {
			double dual_trans_pos = cplex.getDual(costs_pos[i]);
			double dual_trans_neg = cplex.getDual(costs_neg[i]);
			subgrad += dual_trans_pos * value;
			subgrad -= dual_trans_neg * value;
		}
		node->subgradient[i] = subgrad;
	}

	//subgradient for the VaR level
	if(!root_node && (model_->GetParam().risk_measure == RISK_CVAR_MULTIPERIOD)) {
		node->var_subgradient = cplex.getDual(cvar);
		node->var_subgradient_other = cplex.getDual(cvar_other);
	}
	else {
		node->var_subgradient = 0.0;
		node->var_subgradient_other = 0.0;
	}

	//clean up
	env.end();
#endif
}

void SddpSolver::SolveNode(SddpSolverNode *node) {
	if((node->solved_forward_nr == forward_iterations_) && (node->solved_backward_nr == backward_iterations_)) {
		return; //has been solved already in this run
	}
	if((node->solved_forward_nr == forward_iterations_) && (node->tree_node.GetStage() == tree_->StageCount())) {
		return; //no need to solve on the way back because it does not have any cuts
	}

	ptime start_time = microsec_clock::local_time();

	//save some solving time
	if((model_->GetParam().risk_measure == RISK_CVAR_NESTED) && (node->tree_node.GetStage() == model_->GetStagesCount())) {
		//last stage node does not need to be solved
		unsigned int assets = model_->GetAssetsCount();
		//no need for trading costs in the last stage, we just sell or keep all we have
		double total_capital = node->GetCapital(node->parent->solution);
		node->solution[0] = total_capital;
		for(unsigned int i = 1; i < assets; ++i) {
			node->solution[i] = 0.0;
		}
		node->objective = -total_capital;
		node->var = 0.0;
		node->var_other = 0.0;
		for(unsigned int i = 0; i < assets; ++i) {
			node->subgradient[i] = -node->tree_node.GetValues()[i];
		}
		node->var_subgradient = 0.0;
		node->var_subgradient_other = 0.0;
	} else if(EXTERNAL_SOLVER == SOLVER_CPLEX) {
		SolveNodeCplex(node);
	}
	else if(EXTERNAL_SOLVER == SOLVER_COINOR) {
		SolveNodeCoinOr(node);
	}
	else {
		throw SddpSolverException("No external solver provided.");
	}

	//all done
	node->solved_forward_nr = forward_iterations_;
	node->solved_backward_nr = backward_iterations_;
	++solved_lps_;
	ptime end_time = microsec_clock::local_time();
	time_duration comp_time = end_time - start_time;
	lp_solve_time_ += comp_time;
}

void SddpSolver::SolveNodeCoinOr(SddpSolverNode *node) {
	//we build the model using debug wrapper
	CoinModelWrapper * coin_model = new CoinModelWrapper();

	//assets count N
	unsigned int assets = model_->GetAssetsCount();
	unsigned int stage = node->tree_node.GetStage();
	bool root_node = (stage == 1);
	bool last_stage = (stage == model_->GetStagesCount());
	double conf_inv = 1 - model_->GetParam().confidence;
	double conf_inv_other = 1 - model_->GetParam().confidence_other;

	//var x1, ..., xN = weights 1 .. N
	for(unsigned int i = 1; i <= assets; ++i) {
		stringstream str;
		str << "x_" << stage << "_" << i;
		string var_name = str.str();
		coin_model->AddVariable(var_name);
		//weights are positive
		coin_model->AddLowerBound(var_name, 0.0); 
		//max profit = min negative loss
		if(!root_node) {
			switch(model_->GetParam().risk_measure) {
				case RISK_CVAR_NESTED:
					coin_model->AddObjectiveCoefficient(var_name, -1.0);
					break;
				case RISK_CVAR_MULTIPERIOD:
					coin_model->AddObjectiveCoefficient(var_name, -1.0 * expectation_coefficients_[stage - 1]);
					break;
			}
		}
	}

	//var q = lower estimate for the cut
	coin_model->AddVariable("q");
	coin_model->AddLowerBound("q", GetRecourseLowerBound(stage));
	coin_model->AddUpperBound("q", GetRecourseUpperBound(stage));
	//objective with risk aversion coef lambda
	if(!last_stage) {
		//last stage has no recourse
		coin_model->AddObjectiveCoefficient("q",  model_->GetParam().discount_factor);
	}

	//var c = cvar part (positive value)
	if(!root_node && (model_->GetParam().risk_measure == RISK_CVAR_MULTIPERIOD)) {
		coin_model->AddVariable("c");
		coin_model->AddLowerBound("c", 0); //positive part
		//objective with risk aversion coef lambda
		coin_model->AddObjectiveCoefficient("c",  risk_coefficients_[stage-1] / conf_inv);
	}

	//var c_other = cvar part (positive value)
	if(!root_node && (model_->GetParam().risk_measure == RISK_CVAR_MULTIPERIOD)) {
		coin_model->AddVariable("c_other");
		coin_model->AddLowerBound("c_other", 0); //positive part
		//objective with risk aversion coef lambda
		coin_model->AddObjectiveCoefficient("c_other",  risk_coefficients_other_[stage-1] / conf_inv_other);
	}

	//var var = variable to calculate CVaR = VaR level
	coin_model->AddVariable("var");
	coin_model->AddLowerBound("var", GetRecourseLowerBound(stage));
	coin_model->AddUpperBound("var", GetRecourseUpperBound(stage));
	//objevtive according to the risk aversion lambda
	if(!last_stage) {
		//last stage has no recourse
		coin_model->AddObjectiveCoefficient("var", risk_coefficients_[stage]);
	}

	//var var_other = variable to calculate CVaR = VaR level
	coin_model->AddVariable("var_other");
	coin_model->AddLowerBound("var_other", GetRecourseLowerBound(stage));
	coin_model->AddUpperBound("var_other", GetRecourseUpperBound(stage));
	//objevtive according to the risk aversion lambda
	if(!last_stage) {
		//last stage has no recourse
		coin_model->AddObjectiveCoefficient("var_other", risk_coefficients_other_[stage]);
	}

	if(!root_node) {
		//var trans = transaction costs absolute value of stocks position difference
		for(unsigned int i = 1; i <= assets; ++i) {
			stringstream str;
			str << "trans_" << stage << "_" << i;
			string var_name = str.str();
			coin_model->AddVariable(var_name);
			coin_model->AddLowerBound(var_name, 0.0);
		}
	}

	//constraint capital = sum of the weights equals initial capital one
	coin_model->AddConstraint("capital");
	double transaction_coef = model_->GetParam().transaction_costs;
	for(unsigned int i = 1; i <= assets; ++i) {
		stringstream str;
		str << "x_" << stage << "_" << i;
		string var_asset = str.str();
		coin_model->AddConstraintVariable("capital", var_asset, 1);
		if(!root_node) {
			str.str("");
			str << "trans_" << stage << "_" << i;
			string var_absolute = str.str();
			coin_model->AddConstraintVariable("capital", var_absolute, transaction_coef);
		}
	}
	double capital;
	if(root_node) {
		//no parent, init the capital with 1
		capital = 1.0;
	}
	else {
		//sum up the capital under this scenario
		capital = node->GetCapital(node->parent->solution);
	}
	coin_model->AddConstrainBound("capital", EQUAL_TO, capital);

	//constraint the positive part "c" of the CVaR variable
	if(!root_node && (model_->GetParam().risk_measure == RISK_CVAR_MULTIPERIOD)) {
		coin_model->AddConstraint("cvar");
		coin_model->AddConstraintVariable("cvar", "c", 1.0); //the varible itself
		for(unsigned int i = 1; i <= assets; ++i) {
			stringstream str;
			str << "x_" << stage << "_" << i;
			string var_name = str.str();
			coin_model->AddConstraintVariable("cvar", var_name, 1.0); //-c transp x
		}
		coin_model->AddConstrainBound("cvar", GREATER_THAN, -node->parent->var); //previous value at risk
	}

	//constraint the positive part "c_other" of the CVaR variable
	if(!root_node && (model_->GetParam().risk_measure == RISK_CVAR_MULTIPERIOD)) {
		coin_model->AddConstraint("cvar_other");
		coin_model->AddConstraintVariable("cvar_other", "c_other", 1.0); //the varible itself
		for(unsigned int i = 1; i <= assets; ++i) {
			stringstream str;
			str << "x_" << stage << "_" << i;
			string var_name = str.str();
			coin_model->AddConstraintVariable("cvar_other", var_name, 1.0); //-c transp x
		}
		coin_model->AddConstrainBound("cvar_other", GREATER_THAN, -node->parent->var_other); //previous value at risk
	}

	//constraint trans = absolute values for transaction costs
	if(!root_node) { //stage >= 2
		for(unsigned int i = 1; i <= assets; ++i) {
			stringstream str;
			str << "x_" << stage << "_" << i;
			string var_asset = str.str();
			str.str("");
			str << "trans_" << stage << "_" << i;
			string var_absolute = str.str();
			str.str("");
			str << "trans_pos_" << stage << "_" << i;
			string cons_name_pos = str.str();
			str.str("");
			str << "trans_neg_" << stage << "_" << i;
			string cons_name_neg = str.str();
			
			//current position in asset i
			double parent_value = node->tree_node.GetValues()[i-1] * node->parent->solution[i-1];

			//x_t - z <= x_t-1 .. from parent solution
			coin_model->AddConstraint(cons_name_pos);
			coin_model->AddConstraintVariable(cons_name_pos, var_asset, 1);
			coin_model->AddConstraintVariable(cons_name_pos, var_absolute, -1);
			coin_model->AddConstrainBound(cons_name_pos, LOWER_THAN, parent_value);

			//-x_t - z <= -x_t-1 .. from parent solution
			coin_model->AddConstraint(cons_name_neg);
			coin_model->AddConstraintVariable(cons_name_neg, var_asset, -1);
			coin_model->AddConstraintVariable(cons_name_neg, var_absolute, -1);
			coin_model->AddConstrainBound(cons_name_neg, LOWER_THAN, -parent_value);
		}
	}

	//add cuts
	for(unsigned int i = 0; i < cuts_[stage - 1].size(); ++i) {
		SddpSolverCut cut = cuts_[stage - 1][i];
		stringstream str;
		str << "cut_" << stage << "_" << i + 1;
		string constr_name = str.str();
		coin_model->AddConstraint(constr_name);
		coin_model->AddConstraintVariable(constr_name, "q", -1.0);
		for(unsigned int j = 1; j <= assets; ++j) {
			stringstream str_a;
			str_a << "x_" << stage << "_" << j;
			coin_model->AddConstraintVariable(constr_name, str_a.str(), cut.gradient[j - 1]);
		}
		coin_model->AddConstraintVariable(constr_name, "var", cut.gradient[assets]);
		coin_model->AddConstraintVariable(constr_name, "var_other", cut.gradient[assets + 1]);
		coin_model->AddConstrainBound(constr_name, LOWER_THAN, -cut.absolute);
	}

	//solve the model
	coin_model->Solve();

//debug output
#if DEBUG_WRITE_MODEL == 1
	cout << endl << *coin_model << endl;
#endif

	for(unsigned int i = 1; i <= assets; ++i) {
		stringstream str;
		str << "x_" << node->tree_node.GetStage() << "_" << i;
		node->solution[i-1] = coin_model->GetSolution(str.str());
	}

	//fill also the VaR value
	node->var = coin_model->GetSolution("var");

	//fill also the other VaR value
	node->var_other = coin_model->GetSolution("var_other");

	//fill the objectiove
	node->objective = coin_model->GetObjective();

	//calculate subgradient
	//Shapiro SPBook Proposition 2.2 .. T = (-price1, ... -price N) ; W = (1, ..., 1), h = 0
	double dual_cap = coin_model->GetDualPrice("capital");
	for(unsigned int i = 1; i <= assets; ++i) {
		double subgrad = 0.0;
		double value = node->tree_node.GetValues()[i-1];

		//the capital part
		subgrad += dual_cap * value;
		
		//trans costs
		if(!root_node) {
			stringstream str;
			str << "trans_pos_" << stage << "_" << i;
			string cons_name_pos = str.str();
			str.str("");
			str << "trans_neg_" << stage << "_" << i;
			string cons_name_neg = str.str();
			double dual_trans_pos = coin_model->GetDualPrice(cons_name_pos);
			double dual_trans_neg = coin_model->GetDualPrice(cons_name_neg);
			subgrad += dual_trans_pos * value;
			subgrad -= dual_trans_neg * value;
		}

		node->subgradient[i-1] = subgrad;
	}

	//subgradient for the var level
	if(!root_node && (model_->GetParam().risk_measure == RISK_CVAR_MULTIPERIOD)) {
		node->var_subgradient = -coin_model->GetDualPrice("cvar");
		node->var_subgradient_other = -coin_model->GetDualPrice("cvar_other");
	} else {
		node->var_subgradient = 0.0;
		node->var_subgradient_other = 0.0;
	}


	//clean up
	delete coin_model;
}

double SddpSolver::CalculateNodeCapital(SddpSolverNode *node) {
#ifdef _DEBUG
	if(node->parent == 0) {
		throw SddpSolverException("Cannot calculate node value without a parent solution.");
	}
#endif
	return CalculateNodeCapital(node, node->parent);
}

double SddpSolver::CalculateNodeCapital(SddpSolverNode *node, SddpSolverNode *parent) {
#ifdef _DEBUG
	if(parent == 0) {
		throw SddpSolverException("Cannot calculate node value without a parent solution.");
	}
#endif
	return -node->GetCapital(parent->solution);
}

void SddpSolver::AddCut(SddpSolverNode *node) {
#ifdef _DEBUG
	unsigned int count = node->tree_node.GetDescendantsCount();
	if(count < 1) {
		throw SddpSolverException("Cannot make cut with no descendants");
	}
#endif

	if(node->cut_added_nr == backward_iterations_) {
		return; //was processed before
	}

	//prepare the descendant solutions
	ClearNodeDescendants(node);
	ScenarioTreeConstIterator it;
	for(it = node->tree_node.GetDescendantsBegin(); it != node->tree_node.GetDescendantsEnd(); ++it) {
		SddpSolverNode *descnode = BuildNode(*it);
		descnode->parent = node;
		node->descendants.push_back(descnode);
		SolveNode(descnode);
	}

	unsigned int stage = node->tree_node.GetStage();
	double expec_c = expectation_coefficients_[stage];
	double risk_c = risk_coefficients_[stage];
	double risk_c_other = risk_coefficients_other_[stage];
	double conf_inv = 1 - model_->GetParam().confidence;
	double conf_inv_other = 1 - model_->GetParam().confidence_other;
	unsigned int assets = model_->GetAssetsCount();

	//calculate the expected value of the recourse function
	//Shapiro (2001) 4.27
	double q_e = 0;
	double node_val;
	double total_val;
	for(unsigned int i = 0; i < node->descendants.size(); ++i) {
		SddpSolverNode *descendant = node->descendants[i];
		switch(model_->GetParam().risk_measure) {
			case RISK_CVAR_NESTED:
				//expectation and CVaR part
				node_val = descendant->objective;
				total_val = descendant->objective;
				q_e += descendant->GetProbability() * expec_c * node_val;
				if(total_val > node->var) {
					//positive part of CVaR calculation
					q_e += descendant->GetProbability() * (risk_c / conf_inv) * (total_val - node->var);
				}
				if(total_val > node->var_other) {
					//positive part of CVaR calculation
					q_e += descendant->GetProbability() * (risk_c_other / conf_inv_other) * (total_val - node->var_other);
				}
				break;
			case RISK_CVAR_MULTIPERIOD:
				total_val = descendant->objective;
				q_e += descendant->GetProbability() * total_val;
				break;
		}
	}

	//calculate the subgradients
	//Shapiro (2011) 4.29
	colvec subgrad_x = zeros(assets);
	double subgrad_var = 0;
	double subgrad_var_other = 0;
	for(unsigned int i = 0; i < node->descendants.size(); ++i) {
		SddpSolverNode *descendant = node->descendants[i];
		colvec subgrad_desc(descendant->subgradient, assets, false);

		
#if REPORT_NODE_VALUES == 1
		value_file_ << descendant->tree_node.GetStage() <<  "\t" << descendant->objective << endl;
#endif

		switch(model_->GetParam().risk_measure) {
			case RISK_CVAR_NESTED:
				subgrad_x += descendant->GetProbability() * expec_c * subgrad_desc;
				total_val = descendant->objective;
				if(total_val > node->var) {
					subgrad_x += descendant->GetProbability() * (risk_c / conf_inv) * subgrad_desc; 
					subgrad_var -= descendant->GetProbability() * (risk_c / conf_inv);
				}
				if(total_val > node->var_other) {
					subgrad_x += descendant->GetProbability() * (risk_c_other / conf_inv_other) * subgrad_desc; 
					subgrad_var_other -= descendant->GetProbability() * (risk_c_other / conf_inv_other);
				}
			break;
			case RISK_CVAR_MULTIPERIOD:
				subgrad_x += descendant->GetProbability() * subgrad_desc;
				subgrad_var += descendant->GetProbability() * descendant->var_subgradient;
				subgrad_var_other += descendant->GetProbability() * descendant->var_subgradient_other;
			break;
		}
	}
	
	//prepare the cut in the form the gradient colvec and absolute value
	//q(x,var) >= col(1)*x(1) + .. col(N)*x(N) + col(N+1)*var + absolute
	double *cut = new double[assets + 2]; // var & var_other
	for(unsigned int i = 0; i < assets; ++i) {
		cut[i] = subgrad_x(i);
	}
	cut[assets] = subgrad_var;
	cut[assets + 1] = subgrad_var_other;
	//absolute term
	double absolute = q_e;
	for(unsigned int i = 0; i < assets; ++i) {
		absolute -= node->solution[i] * subgrad_x(i);
	}
	absolute -= node->var * subgrad_var;
	absolute -= node->var_other * subgrad_var_other;

	SddpSolverCut scut;
	scut.gradient = cut;
	scut.gradient_size = assets + 2;
	scut.absolute = absolute;
	//append cut
	cuts_[stage - 1].push_back(scut);

	//finalize so that we will not add the same cuts again
	node->cut_added_nr = backward_iterations_;

	//clean up
	ClearNodeDescendants(node);
}

void SddpSolver::CalculateTotalValue(SddpSolverNode *node, double &value, double &probability) {

	if(node->parent == 0) {
		//one deterministic node - should never happen
		throw SddpSolverException("Invalid algorithm state.");
		/*
		int stage = node->tree_node.GetStage();
		return -(expectation_coefficients_mod_[stage-1] + risk_coefficients_mod_[stage-1]) * node->GetValue();
		*/
	}

	//gets the last node int the path and calculates the path cost on the way to parent
	//init the last stage vals
	double exp_value;
	switch(model_->GetParam().risk_measure) {
		case RISK_CVAR_NESTED:
			exp_value = -node->GetValue();
			break;
		case RISK_CVAR_MULTIPERIOD:
			exp_value = -expectation_coefficients_[expectation_coefficients_.size() - 1] * node->GetValue();
			break;
	}
	double recourse_value = exp_value;
	double total_value;
	double node_value;
	double conf_inv = 1 - model_->GetParam().confidence;
	double conf_inv_other = 1 - model_->GetParam().confidence_other;
	double total_probability = (1 / GetConditionalProbability(node)) * GetNodeProbability(node);
	
	//coefficients
	vector<double>::reverse_iterator exp_c = expectation_coefficients_.rbegin();
	vector<double>::reverse_iterator risk_c = risk_coefficients_.rbegin();
	vector<double>::reverse_iterator risk_c_other = risk_coefficients_other_.rbegin();

	//start from the next stage
	SddpSolverNode *last_node = node;
	SddpSolverNode *act_node = node->parent;
	while(act_node != 0) {

#if REPORT_NODE_VALUES == 1
		//if(reported_forward_values_.find(last_node->tree_node.GetIndex()) == reported_forward_values_.end()) {
			value_file_forward_ << last_node->tree_node.GetStage() <<  "\t" << recourse_value << endl;
		//	reported_forward_values_[last_node->tree_node.GetIndex()] = true;
		//}
#endif

		total_value = 0.0;

		switch(model_->GetParam().risk_measure) {
			case RISK_CVAR_NESTED:
				total_value += *exp_c * recourse_value;
				if((recourse_value > act_node->var) && (!last_node->cut_tail)) { 
					//if we can get rid of the tails then do
					total_value += (*risk_c / conf_inv) * (recourse_value - act_node->var);
				}
				if((recourse_value > act_node->var_other) && (!last_node->cut_tail)) { 
					//if we can get rid of the tails then do
					total_value += (*risk_c_other / conf_inv_other) * (recourse_value - act_node->var_other);
				}
				total_value += *risk_c * act_node->var;
				total_value += *risk_c_other * act_node->var_other;
				exp_value = -act_node->GetValue();
				recourse_value = exp_value + total_value;
				++exp_c;
				++risk_c;
				++risk_c_other;
				break;
			case RISK_CVAR_MULTIPERIOD:
				++exp_c; //expcoef is used to calculate past values here..
				total_value += recourse_value;
				node_value = -last_node->GetValue();
				if((node_value > act_node->var) && (!last_node->cut_tail)) { 
					//if we can get rid of the tails then do
					total_value += (*risk_c / conf_inv) * (node_value - act_node->var);
				}
				if((node_value > act_node->var_other) && (!last_node->cut_tail)) { 
					//if we can get rid of the tails then do
					total_value += (*risk_c_other / conf_inv_other) * (node_value - act_node->var_other);
				}
				total_value += *risk_c * act_node->var;
				total_value += *risk_c_other * act_node->var_other;
				exp_value = - *exp_c * act_node->GetValue();
				recourse_value = exp_value + total_value;
				++risk_c;
				++risk_c_other;
				break;
		}
		total_probability *= (1 / GetConditionalProbability(act_node)) * GetNodeProbability(act_node);
	
		last_node = act_node;
		act_node = act_node->parent;
	}

	value = total_value;
	probability = total_probability;
}

void SddpSolver::GetStageSamples(std::vector<unsigned int> &stage_samples) {
	if(model_->GetStagesCount() <= 0) {
		return;
	}
	stage_samples.push_back(1); //fixed root
	//TODO: reconsider
	for(unsigned int stage = 2; stage <= model_->GetStagesCount(); ++stage) {
		if(fix_descendant_count_ > 0) {
			stage_samples.push_back(fix_descendant_count_);
		}
		else {
#if DEBUG_TREE == 1
			stage_samples.push_back(1);
#elif defined _DEBUG
			stage_samples.push_back(20);
#else
			stage_samples.push_back(100);
#endif
		}
	}
}

void SddpSolver::GetReducedSamples(std::vector<unsigned int> &stage_samples) {
	if(model_->GetStagesCount() <= 0) {
		return;
	}
	if(fix_reduced_count_ > 0) {
		stage_samples.push_back(1); //fixed root
		for(unsigned int stage = 2; stage <= model_->GetStagesCount(); ++stage) {
			stage_samples.push_back(fix_reduced_count_);
		}
	}
	else {
		return GetStageSamples(stage_samples);
	}
}

double SddpSolver::GetRecourseLowerBound(unsigned int stage) {
	if(stage == tree_->StageCount()) {
		return 0.0; //last stage has fixed zero recourse
	}
	else {
		return -2.0 * (model_->GetStagesCount() - stage); //TODO: FIX
	}
}

double SddpSolver::GetRecourseUpperBound(unsigned int stage) {
	if(model_->GetParam().risk_measure == RISK_CVAR_NESTED) {
		return 0.0;
	}
	else {
		double conf_inv = 1 - model_->GetParam().confidence_other;
		//get the max lower cvar bound and multiply it by one over atom size
		return -GetRecourseLowerBound(stage) / conf_inv;
	}
}

double SddpSolver::GetNodeProbability(SddpSolverNode *node) {
	if(node->type == NODE_CVAR) {
		return 1 - model_->GetParam().confidence;
	}
	else if(node->type == NODE_EXPECTATION) {
		return model_->GetParam().confidence;
	}
	else if(node->type == NODE_DEFAULT) {
		return 0.5; // we dont know
	}
	else {
		throw SddpSolverException("Invalid node type");
	}
}

double SddpSolver::GetConditionalProbability(SddpSolverNode *node) {
	if(node->type == NODE_CVAR) {
		return GetConditionalProbability(node->tree_node.GetStage());
	}
	else if(node->type == NODE_EXPECTATION) {
		return 1 - GetConditionalProbability(node->tree_node.GetStage());
	}
	else if(node->type == NODE_DEFAULT) {
		return 0.5;
	}
	else {
		throw SddpSolverException("Invalid node type");
	}
}

double SddpSolver::GetConditionalProbability(unsigned int stage) {
	return CONDITIONAL_PROBABILITY;

/*  //5 stage
	if(stage >= 4) {
		return 0.03;
	}
	if(stage >= 3) {
		return 0.08;
	}
	return 0.13;
*/
    //5 stage - coin
/*	if(stage >= 4) {
		return 0.03;
	}
	if(stage >= 3) {
		return 0.08;
	}
	return 0.18;
*/
/*  //10 stage
	if(stage >= 9) {
		return 0.03;
	}
	if(stage >= 8) {
		return 0.06;
	}
	if(stage >= 7) {
		return 0.1;
	}
	if(stage >= 5) {
		return 0.14;
	}
	if(stage >= 4) {
		return 0.25;
	}
	if(stage >= 3) {
		return 0.22;
	}
	return 0.25;
*/
/*
	//15 stage
	if(stage >= 13) {
		return 0.03;
	}
	if(stage >= 12) {
		return 0.05;
	}
	if(stage >= 11) {
		return 0.15;
	}
	if(stage >= 10) {
		return 0.08;
	}
	if(stage >= 9) {
		return 0.15;
	}
	if(stage >= 8) {
		return 0.15;
	}
	if(stage >= 7) {
		return 0.27;
	}
	if(stage >= 6) {
		return 0.58;
	}
	if(stage >= 5) {
		return 0.53;
	}
	if(stage >= 4) {
		return 0.35;
	}
	if(stage >= 3) {
		return 0.58;
	}
	return 0.46;
*/
/*	//15 stage - 1mio
	if(stage >= 13) {
		return 0.03;
	}
	if(stage >= 12) {
		return 0.09;
	}
	if(stage >= 11) {
		return 0.08;
	}
	if(stage >= 10) {
		return 0.2;
	}
	if(stage >= 9) {
		return 0.34;
	}
	if(stage >= 8) {
		return 0.42;
	}
	if(stage >= 7) {
		return 0.77;
	}
	if(stage >= 6) {
		return 0.63;
	}
	if(stage >= 5) {
		return 0.7;
	}
	if(stage >= 4) {
		return 0.71;
	}
	if(stage >= 3) {
		return 0.57;
	}
	return 0.56;
*/
}

SddpSolverNode *SddpSolver::GetRoot() {
	return BuildNode(0);
}