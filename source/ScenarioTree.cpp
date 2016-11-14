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

SCENINDEX TreeNode::GetStageIndex() const {
	return tree_->GetStageIndex(index_);
}


unsigned int TreeNode::GetStage() const {
	return tree_->GetStage(index_);
}

unsigned int TreeNode::GetSize() const {
	return tree_->GetNodeSize();
}
	
const double * TreeNode::GetValues() const {
	return tree_->GetValues(index_);
}

double TreeNode::GetProbability() const {
	return tree_->GetProbability(index_);
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
	
TreeNode TreeNode::GetParent() const {
	return tree_->GetParent(index_);
}

bool TreeNode::HasParent() const {
	return tree_->HasParent(index_);
}

ScenarioTree::ScenarioTree(unsigned int stages, const vector<unsigned int> &stage_samples, unsigned int node_size, StageDependence dependence, 
						   const vector<Distribution*> &stage_distributions, boost::function<void(double *)> evaluate)
{
	stages_ = stages;
	stage_samples_ = stage_samples;
	node_size_ = node_size;
	dependence_ = dependence;
	stage_distributions_ = stage_distributions;
	evaluate_ = evaluate;

	samples_ = 0;
	probabilities_ = 0;

#ifdef _DEBUG
	//ensure data are all right
	if(stage_samples_.size() != stages_) {
		throw ScenarioTreeException("Specified number of stages does not fit specified sample size for each stage.");
	}
	if(stage_distributions_.size() != stages) {
		throw ScenarioTreeException("Specified number of stages does not fit specified distributuions count.");
	}
	if((stages_ > 0) && (stage_samples_[0] != 1)) {
		throw ScenarioTreeException("Root (stage 1) scenario must be deterministic (only 1 scenario).");
	}
#endif
}

ScenarioTree::~ScenarioTree(void)
{
	if(samples_ != 0 || probabilities_ != 0) {
		DestroyTree();
	}
}

SCENINDEX ScenarioTree::ScenarioCount() const {
	return ScenarioCount(stages_);
}

SCENINDEX ScenarioTree::ScenarioCount(unsigned int to_stage) const {
	SCENINDEX count = 0;
	SCENINDEX stage_total = 1;
	for(unsigned int i_stage = 0; i_stage < to_stage; ++i_stage) {
		stage_total *= stage_samples_[i_stage];
		count += stage_total;
	}
	return count;
}

SCENINDEX ScenarioTree::ScenarioSize(unsigned int to_stage) const {
	SCENINDEX count = 0;
	for(unsigned int i_stage = 0; i_stage < to_stage; ++i_stage) {
		count += stage_samples_[i_stage];
	}
	return count;
}

SCENINDEX ScenarioTree::ScenarioCountStage(unsigned int stage) const {
	SCENINDEX stage_total = 1;
	for(unsigned int i_stage = 0; i_stage < stage; ++i_stage) {
		stage_total *= stage_samples_[i_stage];
	}
	return stage_total;
}

unsigned int ScenarioTree::DescendantCountStage(unsigned int stage) const {
	if(stage > stages_) {
		return 0;
	}
	return stage_samples_[stage-1];
}

unsigned int ScenarioTree::StageCount() const {
	return stages_;
}

void ScenarioTree::GenerateTreeIndependent(void) {
	//sample scenarios
	real_count_ = 0;
	samples_ = new mat[stages_];
	probabilities_ = new colvec[stages_];
	for(unsigned int stage = 1; stage <= stages_; ++stage) {
		stage_distributions_[stage-1]->GenerateAnalytic(samples_[stage-1], DescendantCountStage(stage));
		//needed to have colptr access
		samples_[stage-1] = trans(samples_[stage-1]);
		real_count_ += DescendantCountStage(stage);
		if(evaluate_ != 0) {
			for(unsigned int i = 0; i < DescendantCountStage(stage); ++i) {
				evaluate_(samples_[stage-1].colptr(i));
			}
		}
		probabilities_[stage-1].set_size(DescendantCountStage(stage));
		probabilities_[stage-1].fill(1.0 / DescendantCountStage(stage));
	}
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

SCENINDEX ScenarioTree::GetStageIndex(SCENINDEX index) const {
#ifdef _DEBUG
	if(index < 0 || index >= ScenarioCount()) {
		throw ScenarioTreeException("Index out of bounds.");
	}
#endif
	unsigned int stage = GetStage(index);
	return index - ScenarioCount(stage - 1);
}

double * ScenarioTree::GetValues(SCENINDEX index) const {
	unsigned int stage = GetStage(index);
	SCENINDEX scenario_idx = index - ScenarioCount(stage - 1);
	unsigned int real_idx = (scenario_idx % DescendantCountStage(stage)).convert_to<unsigned int>();
	return samples_[stage-1].colptr(real_idx);
}

double ScenarioTree::GetProbability(SCENINDEX index) const {
	unsigned int stage = GetStage(index);
	SCENINDEX scenario_idx = index - ScenarioCount(stage - 1);
	unsigned int real_idx = (scenario_idx % DescendantCountStage(stage)).convert_to<unsigned int>();
	return probabilities_[stage-1](real_idx);
}

void ScenarioTree::SetProbability(SCENINDEX index, double probability) {
#ifdef _DEBUG
	if(probability < 0 || probability > 1) {
		throw ScenarioTreeException("Probability out of bounds.");
	}
#endif
	unsigned int stage = GetStage(index);
	SCENINDEX scenario_idx = index - ScenarioCount(stage - 1);
	unsigned int real_idx = (scenario_idx % DescendantCountStage(stage)).convert_to<unsigned int>();
	probabilities_[stage-1](real_idx) = probability;
}

ScenarioTreeConstIterator ScenarioTree::GetDescendantsBegin(SCENINDEX index) const {
	unsigned int stage = GetStage(index);
	SCENINDEX idx_relative = index - ScenarioCount(stage - 1);
	SCENINDEX idx = ScenarioCount(stage) + idx_relative * DescendantCountStage(stage + 1);
	return ScenarioTreeConstIterator(this, idx);
}

ScenarioTreeConstIterator ScenarioTree::GetDescendantsEnd(SCENINDEX index) const {
	unsigned int stage = GetStage(index);
	SCENINDEX idx_relative = index - ScenarioCount(stage - 1);
	SCENINDEX idx = ScenarioCount(stage) + (idx_relative + 1) * DescendantCountStage(stage + 1);
	return ScenarioTreeConstIterator(this, idx);
}

ScenarioTreeConstIterator ScenarioTree::GetStageBegin(unsigned int stage) const {
	SCENINDEX idx = ScenarioCount(stage - 1);
	return ScenarioTreeConstIterator(this, idx);
}

ScenarioTreeConstIterator ScenarioTree::GetStageEnd(unsigned int stage) const {
	SCENINDEX idx = ScenarioCount(stage);
	return ScenarioTreeConstIterator(this, idx);
}

unsigned int ScenarioTree::GetDescendantsCount(SCENINDEX index) const {
	return DescendantCountStage(GetStage(index) + 1);
}
	
TreeNode ScenarioTree::GetParent(SCENINDEX index) const {
	unsigned int stage = GetStage(index);
#ifdef _DEBUG
	if(stage <= 1) {
		throw ScenarioTreeException("Node has no parent.");
	}
#endif
	SCENINDEX scenario_idx = index - ScenarioCount(stage - 1);
	SCENINDEX parent_idx = scenario_idx / DescendantCountStage(stage);
	return this->operator ()(parent_idx + ScenarioCount(stage - 2));
}

void ScenarioTree::GenerateTree(void) {
	if(samples_ != 0) {
		DestroyTree();
	}

	switch(dependence_) {
		case STAGE_INDEPENDENT:
			GenerateTreeIndependent();
			break;
		default:
			throw ScenarioTreeException("Unknown dependence model between stages."); 
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

unsigned int ScenarioTree::GetNodeSize() const {
	return node_size_;
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
#ifdef _DEBUG
	if(samples_ == 0) {
		throw ScenarioTreeException("Empty tree");
	}
	if(stage < 1 || stage > stages_) {
		throw ScenarioTreeException("Stage out of bounds.");
	}
	if(index < 0 || index >= ScenarioCountStage(stage)) {
		throw ScenarioTreeException("Index out of bounds.");
	}
#endif
	return this->operator ()(ScenarioCount(stage - 1) + index);
}

void ScenarioTree::ReduceSize(const vector<unsigned int> &stage_samples) {
#ifdef _DEBUG
	if(stage_samples.size() != stages_) {
		throw ScenarioTreeException("Specified number of stages does not fit specified sample size for each stage.");
	}
#endif
	for(unsigned int stage = 1; stage <= stages_; ++stage) {
#ifdef _DEBUG
		if(stage_samples[stage - 1] > stage_samples_[stage - 1]) {
			throw ScenarioTreeException("Cannot produce more scenarios, just reduce existing.");
		}
#endif
		while(stage_samples_[stage - 1] > stage_samples[stage - 1]) {
			ReduceOneScenario(stage);
		}
	}
	stage_samples_ = stage_samples;
}

void ScenarioTree::ReduceOneScenario(unsigned int stage) {
	ScenarioTreeConstIterator it_outer;
	ScenarioTreeConstIterator it_inner;
	SCENINDEX idx_delete;
	SCENINDEX idx_target;
	double obj_min = numeric_limits<double>::max();
	//try to find two closest scenarios
	//do this on the first parent, just looping through descendants
	TreeNode parent = GetStageBegin(stage)->GetParent();
	for(it_outer = parent.GetDescendantsBegin(); it_outer != parent.GetDescendantsEnd(); ++it_outer) {
		double distance_min = numeric_limits<double>::max();
		SCENINDEX idx_distance;
		for(it_inner = parent.GetDescendantsBegin(); it_inner != parent.GetDescendantsEnd(); ++it_inner) {
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
	SCENINDEX scenario_idx = idx_delete - ScenarioCount(stage - 1);
	unsigned int real_idx = (scenario_idx % DescendantCountStage(stage)).convert_to<unsigned int>();
	samples_[stage-1].shed_col(real_idx);
	probabilities_[stage-1].shed_row(real_idx);

	//adjust number of elements
	--stage_samples_[stage - 1];
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
