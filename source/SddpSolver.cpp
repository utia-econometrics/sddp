#include "SddpSolver.h"
#include "NormalDistribution.h"
#include "DiscreteDistribution.h"

#include <queue>

SddpSolver::SddpSolver(const ScenarioModel *model, SddpSolverConfig &config)
:Solver(model)
{
	tree_ = 0;
	cuts_.resize(model_->GetStatesCount());
	generator_ = RandomGenerator::GetGenerator();

	config_ = config;

	if(config_.backward_count > config_.forward_count) {
		throw SddpSolverException("Invalid configuration, there have to be more forward pass nodes than backward pass nodes.");
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

	while (forward_iterations_ < config_.max_iterations) {
		if (config_.report_node_values) {
			value_file_.open("value.txt", ios_base::out);
		}

		//go forward and calculate the price
		++forward_iterations_;
		ptime fw_start = microsec_clock::local_time();
		unsigned int fw_start_lps = solved_lps_;
		time_duration fw_start_lp_times = lp_solve_time_;
		cout << "Running forward iteration " << forward_iterations_ << ".. ";

		unsigned int fw_count = config_.forward_count;
		if (config_.debug_bound) {
			if (!do_backward_pass) {
				fw_count = config_.debug_bound_nodes;
			}
		}
		if (config_.solver_strategy == STRATEGY_DEFAULT) {
			//choose the nodes
			ForwardPassStandard(fw_count, forward_nodes);
			upper_bound = CalculateUpperBoundDefault(forward_nodes);
		}
		else if (config_.solver_strategy == STRATEGY_CONDITIONAL) {
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


		if (config_.debug_bound) {
			if (!do_backward_pass && (upper_bounds.size() - lower_bounds.size() >= config_.debug_bound_count)) {
				cout << "Debug bound iteration limit reached, stopping." << endl;
				break;
			}
		}

		//go backward, get the cuts and lower bound
		if (do_backward_pass) {
			++backward_iterations_;
			ptime bw_start = microsec_clock::local_time();
			unsigned int bw_start_lps = solved_lps_;
			time_duration bw_start_lp_times = lp_solve_time_;
			cout << "Running backward iteration " << backward_iterations_ << ".. ";
			//select backward count of nodes
			ChooseRandomSubset(forward_nodes, config_.backward_count, backward_nodes);
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
			if (upper_bounds.size() >= config_.stop_elements) {
				double variance = 0.0;
				ub_mean = 0.0;
				unsigned int elements = 0;
				unsigned int index = upper_bounds.size();
				while (elements < config_.stop_elements) {
					++elements;
					--index;
					ub_mean += upper_bounds[index];
					if (index == 0) {
						break;
					}
				}
				ub_mean /= elements;
				index = upper_bounds.size();
				unsigned int counter = 0;
				while (counter < elements) {
					++counter;
					--index;
					variance += pow(upper_bounds[index] - ub_mean, 2.0);
				}
				variance /= elements;
				NormalDistribution nd(zeros(1), ones(1, 1));
				ub_margin = ub_mean;
				if (config_.stop_use_tests) {
					ub_margin += sqrt(variance) * nd.InverseDistributionFunction(config_.stop_confidence);
				}
				cout << "Stopping margin: " << ub_margin << endl;
				if (!config_.debug_bound) {
					if (ub_margin - lower_bound < config_.convergence_bound) {
						cout << "Upper bound and lower bound difference lower than required precision, stopping." << endl;
						break;
					}
				}
			}

			//need to check upper bounds size also, because of the UB mean calculation
			//add one to the size for stabilization of the upper bound in the case of population mean
			if ((lower_bounds.size() >= config_.stop_elements + 1)
				&& (upper_bounds.size() >= config_.stop_elements + 1)
				&& (lower_bounds[lower_bounds.size() - 1] - lower_bounds[lower_bounds.size() - config_.stop_elements - 1] < config_.epsilon)
				) {
				if (config_.debug_bound) {
					if (config_.stop_cuts == 0) {
						do_backward_pass = false;
					}
				}
				else {
					cout << "Lower bound not improving, stopping." << endl;
					break;
				}
			}

			if (config_.stop_cuts > 0) {
				//no cuts in the last stage, select the ones from the stage before that
				unsigned int idx = tree_->StateCount(tree_->StageCount() - 1);
				if (cuts_[idx - 1].size() >= config_.stop_cuts) {
					do_backward_pass = false;
				}
			}
		}

		//clean up is needed at least where no backward pass is done
		//nodes would remain as descendants and would be counted to upper bound estimators
		ClearNodeDescendants(BuildNode(0));

		if (config_.report_node_values) {
			value_file_.close();
		}

	}

	if (config_.report_node_values) {
		value_file_.close();
	}

	if (forward_iterations_ == config_.max_iterations) {
		cout << "Iteration limit reached, stopping." << endl;
	}

	cout << endl;
	unsigned int index = 0;
	for (; index < lower_bounds.size(); ++index) {
		cout << lower_bounds[index] << " < " << upper_bounds[index] << endl;
	}
	cout << endl;
	for (; index < upper_bounds.size(); ++index) {
		cout << " < " << upper_bounds[index] << endl;
	}
	cout << endl;

	if (config_.report_computation_times) {
		cout << endl;
		cout << "Times:" << endl;
		index = 0;
		for (; index < computation_times_backward_pass.size(); ++index) {
			cout << computation_times_backward_pass[index] << " < " << computation_times_forward_pass[index] << endl;
		}
		cout << endl;
		for (; index < computation_times_forward_pass.size(); ++index) {
			cout << " < " << computation_times_forward_pass[index] << endl;
		}
		cout << endl;
	}

	node = BuildNode(0);
	SolveNode(node);
	weights.reshape(1, model_->GetDecisionSize(1)); //gets size of the first stage = root decision
	for (unsigned int i = 0; i < model_->GetDecisionSize(1); ++i) {
		weights(i) = node->solution[i];
	}
	lower_bound_exact = node->objective;
	upper_bound_mean = ub_mean;
	upper_bound_bound = ub_margin;

	if (config_.debug_bound) {
		double sum = 0;
		unsigned int i = 1;
		for (; i <= config_.debug_bound_count; ++i) {
			if (i > upper_bounds.size()) {
				break;
			}
			sum += upper_bounds[upper_bounds.size() - i];
		}
		unsigned int count = i - 1;
		double mean = sum / count;

		sum = 0;
		for (; i > 0; --i) {
			sum += pow(upper_bounds[upper_bounds.size() - i] - mean, 2);
		}
		double sd = sqrt(sum / (count - 1));
		cout << "Upper bound stats: " << mean << " (" << sd << ")" << endl;

		sum = 0;
		i = 1;
		for (; i <= config_.debug_bound_count; ++i) {
			if (i > computation_times_forward_pass.size()) {
				break;
			}
			sum += computation_times_forward_pass[computation_times_forward_pass.size() - i];
		}
		count = i - 1;
		mean = sum / count;

		sum = 0;
		for (; i > 0; --i) {
			sum += pow(computation_times_forward_pass[computation_times_forward_pass.size() - i] - mean, 2);
		}
		sd = sqrt(sum / (count - 1));
		cout << "Computation time stats [ms]: " << mean << " (" << sd << ")" << endl;
	}
}

void SddpSolver::EvaluatePolicy(boost::function<vector<vector<double> >(vector<const double *>)> policy, double &return_mean, double &return_upper_bound, unsigned int iterations) {
	vector<double> upper_bounds;
	double upper_bound;
	vector<SCENINDEX> forward_nodes;
	unsigned int fw_count = config_.forward_count;
	if (config_.debug_bound) {
		fw_count = config_.debug_bound_nodes;
	}
	for(unsigned int i = 0; i < iterations; ++i){
		//choose the nodes
		if(config_.solver_strategy == STRATEGY_DEFAULT) {
			ForwardPassStandard(fw_count, forward_nodes);
		}
		else if(config_.solver_strategy == STRATEGY_CONDITIONAL) {
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
			vector<vector<double> > solutions = policy(scenario);
			node = base_node;
			index = model_->GetStagesCount() - 1;
			while(node != 0) {
				//full copy of the decisions
				for(unsigned int j = 0; j < node->dimension; ++j) {
					node->solution[j] = solutions[index][j];
				}
				--index;
				node = node->parent;
			}
		}

		//evaluate nodes
		if(config_.solver_strategy == STRATEGY_DEFAULT) {
			//choose the nodes
			upper_bound = CalculateUpperBoundDefault(forward_nodes);
		}
		else if(config_.solver_strategy == STRATEGY_CONDITIONAL) {
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
	double margin = mean;
	if (config_.stop_use_tests) {
		margin += sqrt(variance) * nd.InverseDistributionFunction(config_.stop_confidence);
	}

	return_mean = mean;
	return_upper_bound = margin;
}

vector<vector<double> > SddpSolver::GetPolicy(vector<const double *> scenario) {
	//build a single path tree
	unsigned int stages = model_->GetStagesCount();
	vector<Distribution *> distributions;
	for(unsigned int stage = 1; stage <= stages; ++stage) {
		mat stage_scen(scenario[stage - 1], 1, tree_->GetNodeSize(stage));
		distributions.push_back(new DiscreteDistribution(stage_scen));
	}
	vector<unsigned int> stage_samples;
	for(unsigned int stage = 1; stage <= stages; ++stage) {
		stage_samples.push_back(1); //fixed one exact scenario
	}
	ScenarioTree *tree = new ScenarioTree(stages, stage_samples, STAGE_INDEPENDENT, distributions, 0);
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
		unsigned int decisions = model_->GetDecisionSize(treenode.GetStage());
		node->tree_node = treenode;
		node->solution = new double[decisions];
		node->subgradient = new double[decisions];
		node->dimension = decisions;
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
	vector<vector<double> > solutions;
	for(unsigned int i = 0; i < nodes.size(); ++i) {
		vector<double> stage_soln;
		for(unsigned int d = 0; d < nodes[i]->dimension; ++d) {
			stage_soln.push_back(nodes[i]->solution[d]);
		}
		solutions.push_back(stage_soln);
	}

	//clean up & return
	delete tree;
	for(int i = nodes.size() - 1; i >= 0; --i) {
		delete nodes[i];
	}
	
	return solutions;
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
	unsigned int decisions = model_->GetDecisionSize(treenode.GetStage());
	node->tree_node = treenode;
	node->solution = new double[decisions];
	node->subgradient = new double[decisions];
	node->dimension = decisions;
	node->parent = 0; //explicit initialization
	node->cut_added_nr.resize(treenode.GetNextStatesCount(), 0); //initialize the vector
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
	if(config_.forward_fixed_stage_nodes > 0 && stage_count < config_.forward_fixed_stage_nodes) {
		stage_count = config_.forward_fixed_stage_nodes;
	}

	for(unsigned int stage = 2; stage <= stages; ++stage) {
		actual_nodes.clear();
		if(config_.debug_forward || config_.debug_tree || (config_.quick_conditional && (tree_->DescendantCountTotalStage(stage) <= stage_count))) {
			//choose all nodes
			SCENINDEX from = tree_->operator ()(stage, 1, 0).GetIndex(); //this stage, first state, 0 increment
			SCENINDEX to;
			if (stage == stages) {
				to = tree_->ScenarioCount(); //to the last scenario
			}
			else {
				to = tree_->operator ()(stage + 1, 1, 0).GetIndex(); //next stage, first state, 0 increment
			}
			for(SCENINDEX index = from; index < to; ++index) {
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
					unsigned int next_state = SampleState(parent);
					SddpSolverNode *node = SampleNode(parent, next_state);
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

SddpSolverNode * SddpSolver::SampleNode(SddpSolverNode *parent, unsigned int next_state) {
	double probability = generator_->GetRandom();
	double prop_sum = 0;
	ScenarioTreeConstIterator it;
	SddpSolverNode *node = 0;
	for(it = parent->tree_node.GetDescendantsBegin(next_state); it != parent->tree_node.GetDescendantsEnd(next_state); ++it) {
		prop_sum += it->GetProbability();
		if(prop_sum >= probability) {
			node = BuildNode(*it);
			break;
		}
	}
	return node;
}

unsigned int SddpSolver::SampleState(SddpSolverNode *parent) {
	double probability = generator_->GetRandom();
	double prop_sum = 0;
	unsigned int state = 0;
	ScenarioTreeStateConstIterator it;
	for (it = parent->tree_node.GetNextStatesBegin(); it != parent->tree_node.GetNextStatesEnd(); ++it) {
		prop_sum += it->GetProbability();
		if (prop_sum >= probability) {
			state = it->GetState();
			break;
		}
	}
	return state;
}

void SddpSolver::ForwardPassConditional(unsigned int count, vector<SCENINDEX> &nodes, bool solve) {	
	nodes.clear();
	unsigned int stages = model_->GetStagesCount();
	vector<SddpSolverNode *> actual_nodes;
	vector<SddpSolverNode *> last_nodes;

	//approximate the stage count
	unsigned int stage_count = (count / (stages - 1)) + 1;
	if(config_.forward_fixed_stage_nodes > 0 && stage_count < config_.forward_fixed_stage_nodes) {
		stage_count = config_.forward_fixed_stage_nodes;
	}

	//root
	SddpSolverNode *parent = BuildNode(0);
	SolveNode(parent);
	last_nodes.push_back(parent);

	for(unsigned int stage = 2; stage <= stages; ++stage) {
		actual_nodes.clear();
		if (config_.debug_forward || config_.debug_tree || (config_.quick_conditional && (tree_->DescendantCountTotalStage(stage) <= stage_count))) {
			SCENINDEX from = tree_->operator ()(stage, 1, 0).GetIndex(); //this stage, first state, 0 increment
			SCENINDEX to;
			if (stage == stages) {
				to = tree_->ScenarioCount(); //to the last scenario
			}
			else {
				to = tree_->operator ()(stage + 1, 1, 0).GetIndex(); //next stage, first state, 0 increment
			}
			for (SCENINDEX index = from; index < to; ++index) {
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
				if(parent != last_nodes[parent_idx]) { //optimization when we select multiple nodes for a single parent
					parent = last_nodes[parent_idx];
					unsigned int next_state = SampleState(parent); //sample where we will be in the next state
					nodes.clear();
					ScenarioTreeConstIterator it;
					for(it = parent->tree_node.GetDescendantsBegin(next_state); it != parent->tree_node.GetDescendantsEnd(next_state); ++it) {
						bool existed = NodeExists(*it);
						node = BuildNode(*it);
						//memory optimization - do not connect, calculate value directly using parent and delete
						double value = ApproximateDecisionValue(node, parent);
						nodes.push_back(SddpComparableNode(node->tree_node.GetIndex(), value, node->GetProbability()));
						if(!existed) {
							//only if it was not selected before we can delete it
							ClearSingleNode(node);
						}
					}
					sort(nodes.begin(), nodes.end());

					//marginal node to select
					double tail_cutoff = model_->GetTailAlpha(stage);
					double prop_sum = 0;
					for(unsigned int i = 0; i < nodes.size(); ++i) {
						prop_sum += nodes[i].probability;
						if(prop_sum >= tail_cutoff) {
							margin_index = i;
							margin_probability = prop_sum - nodes[i].probability; //will be incremented by margin index
							double var_h = nodes[i].value;
							cut_margin = model_->CalculateTailCutoff(stage, var_h);
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
					if(config_.cut_nodes_not_tail) {
						if(nodes[selected].value < cut_margin) {
							//we can cut off the downside risk when calculating upper bound
							node->cut_tail = true;
						}
					}
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
	vector<unsigned int> solvestates;
	vector<SCENINDEX> actualnodes;
	vector<unsigned int> actualstates;
	
	//init actual states, only with last stage nodes
	for (unsigned int i = 0; i < nodes.size(); ++i) {
		SddpSolverNode *node = BuildNode(nodes[i]);
#if _DEBUG
		if (node->GetStage() != model_->GetStagesCount()) {
			throw SddpSolverException("Invalid nodes passed to BackwardPass");
		}
#endif
		if (node->parent != 0) {
			actualnodes.push_back(node->parent->tree_node.GetIndex());
			actualstates.push_back(node->GetState());
		}
	}

	while(!actualnodes.empty()) {
		solvenodes.clear();
		solvestates.clear();
		for(unsigned int i = 0; i < actualnodes.size(); ++i) {
			SddpSolverNode *lastnode = BuildNode(actualnodes[i]);
			AddCut(lastnode, actualstates[i]);
			if (lastnode->parent != 0) {
				//will be solved in next round
				solvenodes.push_back(lastnode->parent->tree_node.GetIndex());
				solvestates.push_back(lastnode->GetState());
			}
		}
		actualnodes = solvenodes;
		actualstates = solvestates;
	}
}

void SddpSolver::SolveNode(SddpSolverNode *node) {
	if ((node->solved_forward_nr == forward_iterations_) && (node->solved_backward_nr == backward_iterations_)) {
		return; //has been solved already in this run
	}
	if ((node->solved_forward_nr == forward_iterations_) && (node->GetStage() == tree_->StageCount())) {
		return; //no need to solve on the way back because it does not have any cuts
	}

	ptime start_time = microsec_clock::local_time();

	if (config_.external_solver == SOLVER_CPLEX) {
		SolveNodeCplex(node);
	}
	else if (config_.external_solver == SOLVER_COINOR) {
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

void SddpSolver::SolveNodeCplex(SddpSolverNode *node) {
#ifndef _DEBUG
	//init Ilo environment
	IloEnv env;
	IloModel model(env);
	IloCplex cplex(model);
	IloExpr obj_expr(env);

	//no output
	cplex.setOut(env.getNullStream());

	//Hand over to the model
	unsigned int stage = node->GetStage();
	bool last_stage = (stage == model_->GetStagesCount());
	
	vector<IloNumVar> decision_vars;
	vector<IloRange> dual_constraints;
	
	double *parent_solution = 0;
	if (node->parent != 0) {
		parent_solution = node->parent->solution;
	}
	model_->BuildCplexModel(env, model, obj_expr, stage, parent_solution, node->tree_node.GetValues(), decision_vars, dual_constraints);
	if (decision_vars.size() != model_->GetDecisionSize(stage)) {
		throw SddpSolverException("Size of decision variables vector does not match expected size for this stage.");
	}

	if (!last_stage) {
		//last stage has no recourse
		//var q = lower estimate for the cut
		IloNumVar q(env, GetRecourseLowerBound(stage), GetRecourseUpperBound(stage));
		obj_expr += model_->GetDiscountFactor(stage) * q;
		model.add(q);
		//used to calculate the recourse through multiple descendant states
		IloExpr cut_average(env);
		cut_average += -1 * q; //q = ... average through states ...

		IloRangeArray cuts(env);
		IloNumVarArray q_states(env, node->tree_node.GetNextStatesCount());
		ScenarioTreeStateConstIterator it;
		unsigned int q_idx = 0;
		for (it = node->tree_node.GetNextStatesBegin(); it != node->tree_node.GetNextStatesEnd(); ++it) {
			unsigned int state_idx = it->GetDistributionIndex();
			q_states[q_idx] = IloNumVar(env, GetRecourseLowerBound(stage), GetRecourseUpperBound(stage));
			//add cuts for each state
			for (unsigned int i = 0; i < cuts_[state_idx].size(); ++i) {
				SddpSolverCut cut = cuts_[state_idx][i];
				IloExpr cut_expr(env);
				cut_expr += -1 * q_states[q_idx];
				for (unsigned int j = 0; j < model_->GetDecisionSize(stage); ++j) {
					cut_expr += cut.gradient[j] * decision_vars[j];
				}
				cuts.add(IloRange(env, cut_expr, -cut.absolute));
			}
			
			//bind them with recourse function q
			cut_average += it->GetProbability() * q_states[q_idx];
			++q_idx;
		}

		model.add(q_states);
		cuts.add(IloRange(env, cut_average, 0));
		model.add(cuts);
	}

	//objective
	IloObjective obj = IloMinimize(env, obj_expr);
	model.add(obj);

	if (config_.debug_solver) {
		cplex.exportModel("cplex.lp");
		filesystem::ifstream datafile;
		datafile.open("cplex.lp");
		cout << endl << endl << "Stage: " << stage << endl;
		cout << datafile.rdbuf();
		datafile.close();
		remove("cplex.lp");
	}

	//performance tuning
	cplex.setParam(IloCplex::PreDual, 1);
	cplex.setParam(IloCplex::RootAlg, IloCplex::Primal);

	//solve
	if (!cplex.solve()) {
		cout << "No solution found!" << endl;
		throw SddpSolverException("Solution not found");
	}

	//solutions	
	for (unsigned int i = 0; i < model_->GetDecisionSize(stage); ++i) {
		node->solution[i] = cplex.getValue(decision_vars[i]);
	}

	//fill the objective
	node->objective = cplex.getObjValue();

	if (config_.debug_solver) {
		if (node->parent == 0) {
			cout << "Cplex objective: " << cplex.getObjValue() << endl;
		}
	}

	//get duals
	unsigned int dual_size = dual_constraints.size();
	double *duals = new double[dual_size];
	for (unsigned int i = 0; i < dual_size; ++i) {
		duals[i] = cplex.getDual(dual_constraints[i]);
	}

	//calculate subgradient and save it into node
	model_->FillSubradient(stage, parent_solution, node->tree_node.GetValues(), node->objective, duals, node->recourse, node->subgradient);

	//clean up
	delete[] duals;
	env.end();
#endif // !_DEBUG
}

void SddpSolver::SolveNodeCoinOr(SddpSolverNode *node) {
	//we build the model using debug wrapper
	CoinModelWrapper * coin_model = new CoinModelWrapper();

	//Hand over to the model
	unsigned int stage = node->GetStage();
	bool last_stage = (stage == model_->GetStagesCount());
	vector<string> decision_vars;
	vector<string> dual_constraints;
	double *parent_solution = 0;
	if (node->parent != 0) {
		parent_solution = node->parent->solution;
	}
	model_->BuildCoinModel(coin_model, stage, parent_solution, node->tree_node.GetValues(), decision_vars, dual_constraints);
	if (decision_vars.size() != model_->GetDecisionSize(stage)) {
		throw SddpSolverException("Size of decision variables vector does not match expected size for this stage.");
	}

	if (!last_stage) { //last stage has no recourse
		//var q = lower estimate for the cut
		coin_model->AddVariable("q");
		coin_model->AddObjectiveCoefficient("q", model_->GetDiscountFactor(stage));
		coin_model->AddLowerBound("q", GetRecourseLowerBound(stage));
		coin_model->AddUpperBound("q", GetRecourseUpperBound(stage));
		coin_model->AddConstraint("cut_averaging");
		coin_model->AddConstraintVariable("cut_averaging", "q", -1.0);
		coin_model->AddConstrainBound("cut_averaging", EQUAL_TO, 0);
		
		ScenarioTreeStateConstIterator it;
		for (it = node->tree_node.GetNextStatesBegin(); it != node->tree_node.GetNextStatesEnd(); ++it) {
			unsigned int state_idx = it->GetDistributionIndex();
			//add cuts for each state
			stringstream var_str;
			var_str << "q_" << state_idx;
			string var_name = var_str.str();
			coin_model->AddVariable(var_name);
			for (unsigned int i = 0; i < cuts_[state_idx].size(); ++i) {
				SddpSolverCut cut = cuts_[state_idx][i];
				stringstream str;
				str << "cut_" << state_idx << "_" << i + 1;
				string constr_name = str.str();
				coin_model->AddConstraint(constr_name);
				coin_model->AddConstraintVariable(constr_name, var_name, -1.0);
				for (unsigned int j = 0; j < model_->GetDecisionSize(stage); ++j) {
					coin_model->AddConstraintVariable(constr_name, decision_vars[j], cut.gradient[j]);
				}
				coin_model->AddConstrainBound(constr_name, LOWER_THAN, -cut.absolute);
			}

			//bind them with recourse function q
			coin_model->AddConstraintVariable("cut_averaging", var_name, it->GetProbability());
		}
	}

	//solve the model
	coin_model->Solve();

	//debug output
	if (config_.debug_solver) {
		cout << endl << *coin_model << endl;
	}

	//get decisions
	for(unsigned int i = 0; i < model_->GetDecisionSize(stage); ++i) {
		node->solution[i] = coin_model->GetSolution(decision_vars[i]);
	}

	//fill the objectiove
	node->objective = coin_model->GetObjective();

	//get duals
	unsigned int dual_size = dual_constraints.size();
	double *duals = new double[dual_size];
	for (unsigned int i = 0; i < dual_size; ++i) {
		duals[i] = coin_model->GetDualPrice(dual_constraints[i]);
	}

	//calculate subgradient and save it into node
	model_->FillSubradient(stage, parent_solution, node->tree_node.GetValues(), node->objective, duals, node->recourse, node->subgradient);

	//clean up
	delete[] duals;
	delete coin_model;
}

double SddpSolver::ApproximateDecisionValue(SddpSolverNode *node) {
#ifdef _DEBUG
	if(node->parent == 0) {
		throw SddpSolverException("Cannot calculate node value without a parent solution.");
	}
#endif
	return ApproximateDecisionValue(node, node->parent);
}

double SddpSolver::ApproximateDecisionValue(SddpSolverNode *node, SddpSolverNode *parent) {
#ifdef _DEBUG
	if(parent == 0) {
		throw SddpSolverException("Cannot calculate node value without a parent solution.");
	}
#endif
	return model_->ApproximateDecisionValue(node->GetStage(), parent->solution, node->tree_node.GetValues());
}

void SddpSolver::AddCut(SddpSolverNode *node, unsigned int next_state) {
#ifdef _DEBUG
	unsigned int count = node->tree_node.GetDescendantsCount(next_state);
	if (count < 1) {
		throw SddpSolverException("Cannot make cut with no descendants");
	}
#endif
	
	if (node->cut_added_nr[next_state - 1] == backward_iterations_) {
		return; //was processed before
	}
	
	//prepare the descendant solutions
	ClearNodeDescendants(node);
	ScenarioTreeConstIterator it;
	for (it = node->tree_node.GetDescendantsBegin(next_state); it != node->tree_node.GetDescendantsEnd(next_state); ++it) {
		SddpSolverNode *descnode = BuildNode(*it);
		descnode->parent = node;
		node->descendants.push_back(descnode);
		SolveNode(descnode);
	}

	unsigned int stage = node->GetStage();
	unsigned int dimension = node->dimension;

	//calculate the expected value of the recourse function
	double q_e = 0;
	for (unsigned int i = 0; i < node->descendants.size(); ++i) {
		SddpSolverNode *descendant = node->descendants[i];
		q_e += descendant->GetProbability() * descendant->recourse;
	}

	//calculate the subgradients by averaging over nodes
	colvec subgrad = zeros(dimension);
	for (unsigned int i = 0; i < node->descendants.size(); ++i) {
		SddpSolverNode *descendant = node->descendants[i];
		colvec subgrad_desc(descendant->subgradient, dimension, false);

		if (config_.report_node_values) {
			value_file_ << descendant->tree_node.GetStage() << "\t" << descendant->objective << endl;
		}

		subgrad += descendant->GetProbability() * subgrad_desc;
	}

	//prepare the cut in the form the gradient colvec and absolute value
	//q(x,var) >= col(1)*x(1) + .. col(N)*x(N) + absolute
	double *cut = new double[dimension];
	for (unsigned int i = 0; i < dimension; ++i) {
		cut[i] = subgrad(i);
	}
	//absolute term
	double absolute = q_e;
	for (unsigned int i = 0; i < dimension; ++i) {
		absolute -= node->solution[i] * subgrad(i);
	}

	SddpSolverCut scut;
	scut.gradient = cut;
	scut.gradient_size = dimension;
	scut.absolute = absolute;
	//append cut - it is for the next stage and state is taken from parameter
	cuts_[tree_->GetDistributionIndex(stage + 1, next_state)].push_back(scut);

	//finalize so that we will not add the same cuts again
	node->cut_added_nr[next_state - 1] = backward_iterations_;

	//clean up
	ClearNodeDescendants(node);
}

double SddpSolver::CalculateUpperBoundDefault(const vector<SCENINDEX> &nodes) {
	vector<SddpSolverNode *> parent_nodes;
	vector<SddpSolverNode *> actual_nodes;

	if (config_.report_node_values) {
		value_file_forward_.open("value_forward.txt", ios_base::out);
		reported_forward_values_.clear();
	}

	//fill the parent nodes - we need to average over descendants
	for (unsigned int i = 0; i < nodes.size(); ++i) {
		SddpSolverNode *node = BuildNode(nodes[i]);
#ifdef _DEBUG
		if (node->parent == 0) {
			throw SddpSolverException("Cannot calculate upper bound with no parent nodes.");
		}
#endif
		if (find(actual_nodes.begin(), actual_nodes.end(), node->parent) == actual_nodes.end()) {
			actual_nodes.push_back(node->parent);
		}

		//last stage == no recourse
		node->recourse_value = 0;
	}

	while (actual_nodes.size() > 0) {
		parent_nodes.clear();
		for (unsigned int i = 0; i < actual_nodes.size(); ++i) {

			SddpSolverNode *node = actual_nodes[i];

			if (config_.report_node_values) {
				//if(reported_forward_values_.find(node->tree_node.GetIndex()) == reported_forward_values_.end()) {
				value_file_forward_ << node->GetStage() << "\t" << node->recourse_value << endl;
				//	reported_forward_values_[node->tree_node.GetIndex()] = true;
				//}
			}

			double value = 0.0;
			double probability = 0.0;
			for (unsigned int j = 0; j < node->descendants.size(); ++j) {
				SddpSolverNode *descendant = node->descendants[j];
				//no tail cutting (standard approach)
				double prop_single = descendant->GetProbability() * descendant->tree_node.GetStateProbability();
				value += prop_single * model_->CalculateUpperBound(descendant->GetStage(), node->solution, descendant->solution, descendant->tree_node.GetValues(), descendant->recourse_value, false);
				probability += prop_single;
			}
			value /= probability;
			node->recourse_value = value;

			//add parent for the next iteration
			if (node->parent != 0) {
				if (find(parent_nodes.begin(), parent_nodes.end(), node->parent) == parent_nodes.end()) {
					actual_nodes.push_back(node->parent);
				}
			}
		
		}
		actual_nodes = parent_nodes;
	}

	if (config_.report_node_values) {
		value_file_forward_.close();
	}

	SddpSolverNode *root = BuildNode(0);
	return model_->CalculateUpperBound(root->GetStage(), 0, root->solution, root->tree_node.GetValues(), root->recourse_value, false);
}

double SddpSolver::CalculateUpperBoundConditional(const vector<SCENINDEX> &nodes) {
	double upper_bound = 0.0;
	double total_prop = 0.0;
	if (config_.report_node_values) {
		value_file_forward_.open("value_forward.txt", ios_base::out);
		reported_forward_values_.clear();
	}
	for (unsigned int i = 0; i < nodes.size(); ++i) {
		SddpSolverNode *node = BuildNode(nodes[i]);
		double val;
		double prop;
		CalculateSinglePathUpperBound(node, val, prop);
		upper_bound += prop * val;
		total_prop += prop;
	}
	if (config_.report_node_values) {
		value_file_forward_.close();
	}
	//total_prop adds stability
	upper_bound /= total_prop;
	return upper_bound;
}

void SddpSolver::CalculateSinglePathUpperBound(SddpSolverNode *node, double &value, double &probability) {

	if(node->parent == 0) {
		//one deterministic node - should never happen
		throw SddpSolverException("Invalid algorithm state.");
	}

	//gets the last node in the path and calculates the path cost on the way to parent
	double recourse_value = 0.0;
	double total_probability = 1;
	SddpSolverNode *act_node = node;
	while(act_node != 0) {
		//forward the calculation to the model
		unsigned int stage = act_node->GetStage();
		SddpSolverNode *parent = act_node->parent;
		double *parent_decisions = 0;
		if (parent != 0) {
			parent_decisions = parent->solution;
		}
		recourse_value = model_->CalculateUpperBound(stage, parent_decisions, act_node->solution, act_node->tree_node.GetValues(), recourse_value, act_node->cut_tail);

		//calculate the path probability
		total_probability *= (1 / GetConditionalProbability(act_node)) * GetNodeProbability(act_node) * act_node->GetProbability() * act_node->tree_node.GetStateProbability();

		if (config_.report_node_values) {
			//if(reported_forward_values_.find(last_node->tree_node.GetIndex()) == reported_forward_values_.end()) {
			value_file_forward_ << act_node->GetStage() << "\t" << recourse_value << endl;
			//	reported_forward_values_[last_node->tree_node.GetIndex()] = true;
			//}
		}

		//go up!
		act_node = parent;
	}

	value = recourse_value;
	probability = total_probability;
}

void SddpSolver::GetStageSamples(std::vector<unsigned int> &stage_samples) {
	if(model_->GetStagesCount() <= 0) {
		return;
	}
	stage_samples.push_back(1); //fixed root

	for(unsigned int stage = 2; stage <= model_->GetStagesCount(); ++stage) {
		if(config_.samples_per_stage > 0) {
			stage_samples.push_back(config_.samples_per_stage);
		}
		else {
			if (config_.debug_tree) {
				stage_samples.push_back(1);
			}
			else {
				stage_samples.push_back(100); //Solver default
			}
		}
	}
}

void SddpSolver::GetReducedSamples(std::vector<unsigned int> &stage_samples) {
	if(model_->GetStagesCount() <= 0) {
		return;
	}
	if(config_.reduced_samples_per_stage > 0) {
		stage_samples.push_back(1); //fixed root
		for(unsigned int stage = 2; stage <= model_->GetStagesCount(); ++stage) {
			stage_samples.push_back(config_.reduced_samples_per_stage);
		}
	}
	else {
		return GetStageSamples(stage_samples);
	}
}

double SddpSolver::GetRecourseLowerBound(unsigned int stage) {
	return model_->GetRecourseLowerBound(stage);
}

double SddpSolver::GetRecourseUpperBound(unsigned int stage) {
	return model_->GetRecourseUpperBound(stage);
}

double SddpSolver::GetNodeProbability(SddpSolverNode *node) {
	if(node->type == NODE_CVAR) {
		return 1 - model_->GetTailAlpha(node->GetStage());
	}
	else if(node->type == NODE_EXPECTATION) {
		return model_->GetTailAlpha(node->GetStage());
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
		return GetConditionalProbability(node->GetStage());
	}
	else if(node->type == NODE_EXPECTATION) {
		return 1 - GetConditionalProbability(node->GetStage());
	}
	else if(node->type == NODE_DEFAULT) {
		return 0.5;
	}
	else {
		throw SddpSolverException("Invalid node type");
	}
}

double SddpSolver::GetConditionalProbability(unsigned int stage) {
	return config_.conditional_probability;
}

SddpSolverNode *SddpSolver::GetRoot() {
	return BuildNode(0);
}