#include "ScenarioTree.h"
#include <queue>
#include <boost/bind.hpp>
#include <limits>

using namespace boost;

TreeNode::TreeNode(const ScenarioTree *tree, SCENINDEX index) {
	tree_ = tree;
	index_ = index;
}

TreeNode::TreeNode() {
	tree_ = 0;
	index_ = 0;
}

TreeNode::~TreeNode() {
}

SCENINDEX TreeNode::GetNumber() const {
	return index_ + 1;
}

SCENINDEX TreeNode::GetIndex() const {
	return index_;
}


unsigned int TreeNode::GetStage() const {
	return tree_->GetStage(index_);
}

unsigned int TreeNode::GetState() const {
	return tree_->GetState(index_);
}

unsigned int TreeNode::GetSize() const {
	return tree_->GetNodeSize(tree_->GetStage(index_));
}
	
const double * TreeNode::GetValues() const {
	return tree_->GetValues(index_);
}

double TreeNode::GetProbability() const {
	return tree_->GetProbability(index_);
}

double TreeNode::GetStateProbability() const {
	if (HasParent()) {
		return tree_->GetStateProbability(GetParent().GetIndex(), GetState());
	}
	else {
		//root
		return 1.0;
	}
}

ScenarioTreeConstIterator TreeNode::GetDescendantsBegin() const {
	return tree_->GetDescendantsBegin(index_);
}

ScenarioTreeConstIterator TreeNode::GetDescendantsEnd() const {
	return tree_->GetDescendantsEnd(index_);
}
	
unsigned int TreeNode::GetDescendantsCount() const {
	return tree_->GetDescendantsCount(index_);
}

ScenarioTreeConstIterator TreeNode::GetDescendantsBegin(unsigned int next_state) const {
	return tree_->GetDescendantsBegin(index_, next_state);
}

ScenarioTreeConstIterator TreeNode::GetDescendantsEnd(unsigned int next_state) const {
	return tree_->GetDescendantsEnd(index_, next_state);
}

unsigned int TreeNode::GetDescendantsCount(unsigned int next_state) const {
	return tree_->GetDescendantsCount(index_, next_state);
}
	
TreeNode TreeNode::GetParent() const {
	return tree_->GetParent(index_);
}

bool TreeNode::HasParent() const {
	return tree_->HasParent(index_);
}

unsigned int TreeNode::GetDistributionIndex() const {
	return tree_->GetDistributionIndex(index_);
}

ScenarioTreeStateConstIterator TreeNode::GetNextStatesBegin() const {
	return tree_->GetNextStatesBegin(index_);
}

ScenarioTreeStateConstIterator TreeNode::GetNextStatesEnd() const {
	return tree_->GetNextStatesEnd(index_);
}

unsigned int TreeNode::GetNextStatesCount() const {
	return tree_->StateCountStage(GetStage() + 1);
}

TreeState::TreeState(const ScenarioTree *tree, SCENINDEX parent_index, unsigned int state) {
	tree_ = tree;
	parent_index_ = parent_index;
	state_ = state;
}

TreeState::TreeState() {
	tree_ = 0;
	parent_index_ = 0;
	state_ = 0;
}

TreeState::~TreeState() {
}

double TreeState::GetProbability() const {
	return tree_->GetStateProbability(parent_index_, state_);
}

unsigned int TreeState::GetStage() const {
	return tree_->GetStage(parent_index_) + 1; //we are one stage from parent
}

unsigned int TreeState::GetState() const {
	return state_;
}

unsigned int TreeState::GetDistributionIndex() const {
	return tree_->GetDistributionIndex(GetStage(), GetState());
}

ScenarioTree::ScenarioTree(unsigned int stages, const vector<unsigned int> &stage_samples, StageDependence dependence,
						   const vector<Distribution*> &stage_distributions, boost::function<void(double *)> evaluate)
{
	if (dependence != STAGE_INDEPENDENT) {
		throw ScenarioTreeException("Invalid constructor call, use other versions");
	}

	//convert one stage per stage to vectors to map to general version
	vector<vector<unsigned int> > all_stage_samples;
	for (unsigned int stage = 1; stage <= stages; ++stage) {
		vector<unsigned int> one_stage_samples;
		one_stage_samples.push_back(stage_samples[stage - 1]);
		all_stage_samples.push_back(one_stage_samples);
	}

	vector<vector<Distribution*> > all_stage_distributions;
	for (unsigned int stage = 1; stage <= stages; ++stage) {
		vector<Distribution*> one_stage_distribution;
		one_stage_distribution.push_back(stage_distributions[stage - 1]);
		all_stage_distributions.push_back(one_stage_distribution);
	}

	vector<mat> transition_probabilities;
	for (unsigned int stage = 1; stage < stages; ++stage) { //from last state there is no transition
		mat prob = ones(1); //just one state, transition to it is sure
		transition_probabilities.push_back(prob);
	}

	vector<unsigned int> state_counts;
	for (unsigned int stage = 1; stage <= stages; ++stage) {
		state_counts.push_back(1); //just one state in all stages
	}

	Init(stages, state_counts, all_stage_samples, dependence, all_stage_distributions, transition_probabilities, evaluate);
}

ScenarioTree::ScenarioTree(unsigned int stages, vector<unsigned int> state_counts, 
	const vector<vector<unsigned int> > &stage_samples, StageDependence dependence,
	const vector<vector<Distribution*> > &stage_distributions, vector<mat> transition_probabilities,
	boost::function<void(double *)> evaluate) {
	Init(stages, state_counts, stage_samples, dependence, stage_distributions, transition_probabilities, evaluate);
}

void ScenarioTree::Init(unsigned int stages, vector<unsigned int> state_counts,
	const vector<vector<unsigned int> > &stage_samples, StageDependence dependence,
	const vector<vector<Distribution*> > &stage_distributions, vector<mat> transition_probabilities,
	boost::function<void(double *)> evaluate) {
	stages_ = stages;
	state_counts_ = state_counts;
	stage_samples_ = stage_samples;
	dependence_ = dependence;
	stage_distributions_ = stage_distributions;
	transition_probabilities_ = transition_probabilities;
	evaluate_ = evaluate;

	samples_ = 0;
	probabilities_ = 0;

#ifdef _DEBUG
	//ensure data are all right
	if (stage_samples_.size() != stages_) {
		throw ScenarioTreeException("Specified number of stages does not fit specified sample size for each stage.");
	}
	if (stage_distributions_.size() != stages) {
		throw ScenarioTreeException("Specified number of stages does not fit specified distributuions count.");
	}
	if ((stages_ > 0) && (stage_samples_[0][0] != 1)) {
		throw ScenarioTreeException("Root (stage 1) scenario must be deterministic (only 1 scenario).");
	}
	for (unsigned int stage = 1; stage <= stages; ++stage) {
		if (stage_samples_[stage - 1].size() != stage_distributions_[stage - 1].size()) {
			throw ScenarioTreeException("State counts in sampling size vector and distribution vector do not match");
		}
	}
	for (unsigned int stage = 1; stage <= stages; ++stage) {
		if (stage_distributions_[stage - 1].size() != state_counts_[stage - 1]) {
			throw ScenarioTreeException("State counts provided do not match distributions vector");
		}
	}
	if (transition_probabilities_.size() != stages - 1) {
		throw ScenarioTreeException("Incorrect size of transition probabilities vector");
	}
	for (unsigned int stage_from = 1; stage_from < stages; ++stage_from) { //just to stages - 1 since we need transtition to
		unsigned int stage_to = stage_from + 1;
		if (transition_probabilities_[stage_from - 1].n_rows != state_counts_[stage_from - 1]) {
			throw ScenarioTreeException("Transition matrix row count does not equal to number of states in starting stage");
		}
		if (transition_probabilities_[stage_from - 1].n_cols != state_counts_[stage_to - 1]) {
			throw ScenarioTreeException("Transition matrix column count does not equal to number of states in ending stage");
		}
	}
#endif

	//fill out node sizes from distributions
	for (unsigned int stage = 1; stage <= stages; ++stage) {
		unsigned int size = stage_distributions_[stage - 1][0]->GetDimension(); //first size == default
		stage_node_size_.push_back(size);
#ifdef _DEBUG
		for (unsigned int idx = 1; idx < stage_distributions_[stage - 1].size(); ++idx) { //start from second member of array
			if (stage_distributions_[stage - 1][idx]->GetDimension() != size) {
				throw ScenarioTreeException("Dimension of random vectors for different states in one stage do not match");
			}
		}
#endif
	}
}

ScenarioTree::~ScenarioTree(void)
{
	if(samples_ != 0 || probabilities_ != 0) {
		DestroyTree();
	}
}

SCENINDEX ScenarioTree::ScenarioCount() const {
	return ScenarioCountToStage(stages_);
}

SCENINDEX ScenarioTree::ScenarioCountToStage(unsigned int to_stage) const {
	SCENINDEX count = 0;
	for(unsigned int stage = 1; stage <= to_stage; ++stage) {
		count += ScenarioCountStage(stage);
	}
	return count;
}
/* if to_state == 0 then it is the count to to_stage - 1 */
SCENINDEX ScenarioTree::ScenarioCountToState(unsigned int to_stage, unsigned int to_state) const {
	SCENINDEX count = ScenarioCountToStage(to_stage - 1);
	for (unsigned int state = 1; state <= to_state; ++state) {
		count += ScenarioCountState(to_stage, state);
	}
	return count;
}

unsigned int ScenarioTree::StateCount(unsigned int to_stage) const {
	unsigned int count = 0;
	for (unsigned int stage = 1; stage <= to_stage; ++stage) {
		count += StateCountStage(stage);
	}
	return count;
}

SCENINDEX ScenarioTree::ScenarioCountStage(unsigned int stage) const {
	SCENINDEX stage_total = 1;
	for (unsigned int i_stage = 0; i_stage < stage; ++i_stage) {
		SCENINDEX one_stage_count = 0;
		for (unsigned int i_state = 0; i_state < stage_samples_[i_stage].size(); ++i_state) {
			one_stage_count += stage_samples_[i_stage][i_state];
		}
		stage_total *= one_stage_count;
	}
	return stage_total;
}

SCENINDEX ScenarioTree::ScenarioCountState(unsigned int stage, unsigned int state) const {
	return ScenarioCountStage(stage - 1) * stage_samples_[stage - 1][state - 1];
}

unsigned int ScenarioTree::DescendantCountStage(unsigned int stage) const {
	if (dependence_ == STAGE_INDEPENDENT) {
		return DescendantCountStage(stage, 1); //fix one state = 0
	}
	else {
		throw ScenarioTreeException("Call to undefined functions without specification of state");
	}
}

unsigned int ScenarioTree::StateCountStage(unsigned int stage) const {
	return state_counts_[stage - 1];
}

unsigned int ScenarioTree::DescendantCountTotalStage(unsigned int stage) const {
	unsigned int total = 0;
	for (unsigned int state = 1; state <= StateCountStage(stage); ++state) {
		total += DescendantCountStage(stage, state);
	}
	return total;
}

unsigned int ScenarioTree::DescendantCountStage(unsigned int stage, unsigned int state) const {
	if (stage > stages_) {
		return 0;
	}
	return stage_samples_[stage - 1][state - 1];
}

unsigned int ScenarioTree::StageCount() const {
	return stages_;
}

unsigned int ScenarioTree::GetStage(SCENINDEX index) const {
#ifdef _DEBUG
	if(index < 0 || index >= ScenarioCount()) {
		throw ScenarioTreeException("Index out of bounds.");
	}
#endif
	SCENINDEX counter = 0;
	unsigned int stage = 0;
	while(counter <= index) {
		counter += ScenarioCountStage(++stage);
	}
	return stage;
}

unsigned int ScenarioTree::GetState(SCENINDEX index) const {
	unsigned int stage = GetStage(index);
	SCENINDEX scenario_idx = index - ScenarioCountToStage(stage - 1);
	SCENINDEX counter = 0;
	unsigned int state = 0;
	while (counter <= scenario_idx) {
		++state;
		counter += ScenarioCountState(stage, state);
	}
	return state;
}

unsigned int ScenarioTree::GetSampleIndex(SCENINDEX index) const {
	unsigned int stage = GetStage(index);
	unsigned int state = GetState(index);

	SCENINDEX scenario_idx = index - ScenarioCountToState(stage, state - 1);
	unsigned int real_idx = (scenario_idx % DescendantCountStage(stage, state)).convert_to<unsigned int>();
	return real_idx;
}

double * ScenarioTree::GetValues(SCENINDEX index) const {
	unsigned int distr_idx = GetDistributionIndex(index);
	unsigned int sampl_idx = GetSampleIndex(index);
	return samples_[distr_idx].colptr(sampl_idx);
}

double ScenarioTree::GetProbability(SCENINDEX index) const {
	unsigned int distr_idx = GetDistributionIndex(index);
	unsigned int sampl_idx = GetSampleIndex(index);
	return probabilities_[distr_idx](sampl_idx);
}

void ScenarioTree::SetProbability(SCENINDEX index, double probability) {
#ifdef _DEBUG
	if(probability < 0 || probability > 1) {
		throw ScenarioTreeException("Probability out of bounds.");
	}
#endif
	unsigned int distr_idx = GetDistributionIndex(index);
	unsigned int sampl_idx = GetSampleIndex(index);
	probabilities_[distr_idx](sampl_idx) = probability;
}

ScenarioTreeConstIterator ScenarioTree::GetDescendantsBegin(SCENINDEX index) const {
	if (dependence_ == STAGE_INDEPENDENT) {
		return GetDescendantsBegin(index, 1); //fix one state = 1
	}
	else {
		throw ScenarioTreeException("Call to undefined functions without specification of state");
	}
}

ScenarioTreeConstIterator ScenarioTree::GetDescendantsEnd(SCENINDEX index) const {
	if (dependence_ == STAGE_INDEPENDENT) {
		return GetDescendantsEnd(index, 1); //fix one state = 1
	}
	else {
		throw ScenarioTreeException("Call to undefined functions without specification of state");
	}
}

unsigned int ScenarioTree::GetDescendantsCount(SCENINDEX index) const {
	if (dependence_ == STAGE_INDEPENDENT) {
		return GetDescendantsCount(index, 1); //fix one state = 1
	}
	else {
		throw ScenarioTreeException("Call to undefined functions without specification of state");
	}
}

ScenarioTreeConstIterator ScenarioTree::GetDescendantsBegin(SCENINDEX index, unsigned int next_state) const {
	unsigned int stage = GetStage(index);
	unsigned int state = GetState(index);
	SCENINDEX idx_relative = index - ScenarioCountToStage(stage - 1);
	SCENINDEX idx = ScenarioCountToState(stage + 1, next_state - 1) + idx_relative * DescendantCountStage(stage + 1, next_state);
	return ScenarioTreeConstIterator(this, idx);
}

ScenarioTreeConstIterator ScenarioTree::GetDescendantsEnd(SCENINDEX index, unsigned int next_state) const {
	unsigned int stage = GetStage(index);
	unsigned int state = GetState(index);
	SCENINDEX idx_relative = index - ScenarioCountToStage(stage - 1);
	SCENINDEX idx = ScenarioCountToState(stage + 1, next_state - 1) + (idx_relative + 1) * DescendantCountStage(stage + 1, next_state);
	return ScenarioTreeConstIterator(this, idx);
}

unsigned int ScenarioTree::GetDescendantsCount(SCENINDEX index, unsigned int next_state) const {
	return DescendantCountStage(GetStage(index) + 1, next_state);
}
	
TreeNode ScenarioTree::GetParent(SCENINDEX index) const {
	unsigned int stage = GetStage(index);
	unsigned int state = GetState(index);
#ifdef _DEBUG
	if(stage <= 1) {
		throw ScenarioTreeException("Node has no parent.");
	}
#endif
	SCENINDEX scenario_idx = index - ScenarioCountToState(stage, state - 1);
	SCENINDEX parent_idx = scenario_idx / DescendantCountStage(stage, state);
	return this->operator ()(parent_idx + ScenarioCountToStage(stage - 2));
}

unsigned int ScenarioTree::GetDistributionIndex(SCENINDEX index) const {
	unsigned int stage = GetStage(index);
	unsigned int state = GetState(index);
	return GetDistributionIndex(stage, state);
}

unsigned int ScenarioTree::GetDistributionIndex(unsigned int stage, unsigned int state) const {
	unsigned int idx = StateCount(stage - 1); //index of this stage starts here
	idx += (state - 1); //real scenario increment
	return idx;
}

double ScenarioTree::GetStateProbability(SCENINDEX index, unsigned int next_state) const {
	if (dependence_ == STAGE_INDEPENDENT) {
		return 1.0; //no states
	}
	else if (dependence_ == MARKOV) {
		unsigned int stage = GetStage(index);
		unsigned int state = GetState(index);
		return transition_probabilities_[stage - 1](state - 1, next_state - 1);
	}
	throw ScenarioTreeException("Undefined dependence system");
}

ScenarioTreeStateConstIterator ScenarioTree::GetNextStatesBegin(SCENINDEX index) const {
	return ScenarioTreeStateConstIterator(this, index, 1);
}

ScenarioTreeStateConstIterator ScenarioTree::GetNextStatesEnd(SCENINDEX index) const {
	unsigned int stage = GetStage(index);
	return ScenarioTreeStateConstIterator(this, index, StateCountStage(stage + 1) + 1);
}


void ScenarioTree::GenerateTree(void) {
	if(samples_ != 0) {
		DestroyTree();
	}

	//sample scenarios
	unsigned int total_count = StateCount(stages_);
	samples_ = new mat[total_count];
	probabilities_ = new colvec[total_count];
	for (unsigned int stage = 1; stage <= stages_; ++stage) {
		for (unsigned int state = 1; state <= StateCountStage(stage); ++state) {
			unsigned int index = GetDistributionIndex(stage, state);
			stage_distributions_[stage - 1][state - 1]->GenerateAnalytic(samples_[index], DescendantCountStage(stage, state));
			//needed to have colptr access
			samples_[index] = trans(samples_[index]);
			if (evaluate_ != 0) {
				for (unsigned int i = 0; i < DescendantCountStage(stage, state); ++i) {
					evaluate_(samples_[index].colptr(i));
				}
			}
			probabilities_[index].set_size(DescendantCountStage(stage, state));
			probabilities_[index].fill(1.0 / DescendantCountStage(stage, state));
		}
	}
}

void ScenarioTree::DestroyTree() {
	//delete samples and their probabilities
	delete[] samples_;
	delete[] probabilities_;
}

TreeNode ScenarioTree::GetRoot() const {
	return this->operator ()(0);
}

unsigned int ScenarioTree::GetNodeSize(unsigned int stage) const {
	return stage_node_size_[stage - 1];;
}

bool ScenarioTree::HasParent(SCENINDEX index) const {
	return !(index == 0);
}

TreeNode ScenarioTree::operator() (SCENINDEX index) const {
#ifdef _DEBUG
	if(samples_ == 0) {
		throw ScenarioTreeException("Empty tree");
	}
	if(index < 0 || index >= ScenarioCount()) {
		throw ScenarioTreeException("Index out of bounds.");
	}
#endif
	return TreeNode(this, index);
}

TreeNode ScenarioTree::operator() (unsigned int stage, SCENINDEX index) const {
	if (dependence_ == STAGE_INDEPENDENT) {
		return this->operator ()(stage, 1, index); //fix one state = 1
	}
	else {
		throw ScenarioTreeException("Call to undefined functions without specification of state");
	}
}

TreeNode ScenarioTree::operator() (unsigned int stage, unsigned int state, SCENINDEX index) const {
#ifdef _DEBUG
	if(samples_ == 0) {
		throw ScenarioTreeException("Empty tree");
	}
	if(stage < 1 || stage > stages_) {
		throw ScenarioTreeException("Stage out of bounds.");
	}
	if (state < 1 || state > StateCountStage(stage)) {
		throw ScenarioTreeException("State out of bounds.");
	}
	if(index < 0 || index >= ScenarioCountState(stage, state)) {
		throw ScenarioTreeException("Index out of bounds.");
	}
#endif
	return this->operator ()(ScenarioCountToState(stage, state - 1) + index);
}

void ScenarioTree::ReduceSize(const vector<unsigned int> &stage_samples) {
	//TODO: reconsider full generality here
	//convert one stage per stage to vectors to map to general version
	vector<vector<unsigned int> > all_stage_samples;
	for (unsigned int stage = 1; stage <= stages_; ++stage) {
		vector<unsigned int> one_stage_samples;
		one_stage_samples.push_back(stage_samples[stage - 1]);
		all_stage_samples.push_back(one_stage_samples);
	}
}

void ScenarioTree::ReduceSize(const vector<vector<unsigned int> > &stage_samples) {
#ifdef _DEBUG
	if(stage_samples.size() != stages_) {
		throw ScenarioTreeException("Specified number of stages does not fit specified sample size for each stage.");
	}
#endif
	for(unsigned int stage = 1; stage <= stages_; ++stage) {
		for (unsigned int state = 1; state <= StateCountStage(stage); ++state) {
#ifdef _DEBUG
			if (stage_samples[stage - 1][state - 1] > stage_samples_[stage - 1][state - 1]) {
				throw ScenarioTreeException("Cannot produce more scenarios, just reduce existing.");
			}
#endif
			while (stage_samples_[stage - 1][state - 1] > stage_samples[stage - 1][state - 1]) {
				ReduceOneScenario(stage, state);
			}
		}
	}
	stage_samples_ = stage_samples;
}

void ScenarioTree::ReduceOneScenario(unsigned int stage, unsigned int state) {
	ScenarioTreeConstIterator it_outer;
	ScenarioTreeConstIterator it_inner;
	SCENINDEX idx_delete;
	SCENINDEX idx_target;
	double obj_min = numeric_limits<double>::max();
	//try to find two closest scenarios
	//do this on the first parent, just looping through descendants
	TreeNode current = this->operator ()(stage, state, 0);
	TreeNode parent = current.GetParent();
	for(it_outer = parent.GetDescendantsBegin(state); it_outer != parent.GetDescendantsEnd(state); ++it_outer) {
		double distance_min = numeric_limits<double>::max();
		SCENINDEX idx_distance;
		for(it_inner = parent.GetDescendantsBegin(state); it_inner != parent.GetDescendantsEnd(state); ++it_inner) {
			if(it_outer == it_inner) {
				continue;
			}
			double distance = ScenarioDistance(it_outer->GetIndex(), it_inner->GetIndex());
			if(distance < distance_min) {
				distance_min = distance;
				idx_distance = it_inner->GetIndex();
			}
		}
		double obj = it_outer->GetProbability() * distance_min;
		if(obj < obj_min) {
			obj_min = obj;
			idx_delete = it_outer->GetIndex();
			idx_target = idx_distance;
		}
	}

	//now redistribute probability and delete
	SetProbability(idx_target, GetProbability(idx_target) + GetProbability(idx_delete));
	unsigned int distr_idx = GetDistributionIndex(stage, state);
	unsigned int sampl_idx = GetSampleIndex(idx_delete);
	samples_[distr_idx].shed_col(sampl_idx);
	probabilities_[distr_idx].shed_row(sampl_idx);

	//adjust number of elements
	--stage_samples_[stage - 1][state - 1];
}

double ScenarioTree::ScenarioDistance(SCENINDEX s1, SCENINDEX s2) const {
	TreeNode t1 = this->operator()(s1);
	TreeNode t2 = this->operator()(s2);
#ifdef _DEBUG
	if(t1.GetSize() != t2.GetSize()) {
		throw ScenarioTreeException("Cannot compute distance of two vectors with unequal dimensions.");
	}
#endif
	
	double sum = 0.0;
	const double *values1 = t1.GetValues();
	const double *values2 = t2.GetValues();
	for(unsigned int i = 0; i < t1.GetSize(); ++i) {
		sum += pow(values1[i] - values2[i], 2);
	}
	return sqrt(sum);
}