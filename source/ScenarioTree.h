#pragma once

#include "Distribution.h"
#include "Exception.h"
#include <boost/function.hpp>
#include "Helper.h"
#include <boost/iterator/iterator_facade.hpp>

using namespace std;

class ScenarioTree; //forward
class ScenarioTreeConstIterator; //forward
class ScenarioTreeStateConstIterator; //forward

//runtime exception
class ScenarioTreeException : Exception
{ 
public:
	ScenarioTreeException(string text) : Exception(text){
	}
};

enum StageDependence {
	STAGE_INDEPENDENT = 0,
	MARKOV = 1
	/* to be implemented */
};

class TreeNode {
public:
	TreeNode(const ScenarioTree *tree, SCENINDEX index);
	TreeNode();
	~TreeNode();

	SCENINDEX GetNumber() const; //nubmer of the scenario
	unsigned int GetStage() const; //stage of the scenario
	unsigned int GetState() const; //returns the status of the Markov Chain
	const double * GetValues() const; //prices of assets
	double GetProbability() const; //probability of this scenario
	double GetStateProbability() const; //probability of this scenario
	ScenarioTreeConstIterator GetDescendantsBegin() const; //following stages
	ScenarioTreeConstIterator GetDescendantsEnd() const; //following stages
	unsigned int GetDescendantsCount() const; //scenarios number in next stage
	ScenarioTreeConstIterator GetDescendantsBegin(unsigned int next_state) const; //following stages
	ScenarioTreeConstIterator GetDescendantsEnd(unsigned int next_state) const; //following stages
	unsigned int GetDescendantsCount(unsigned int next_state) const; //scenarios number in next stage
	TreeNode GetParent() const; //pointer to the parent
	bool HasParent() const; //true if has parent
	SCENINDEX GetIndex() const; //true index
	unsigned int GetSize() const; //number of values in the node
	unsigned int GetDistributionIndex() const; //returns the indexing position of real distribution
	ScenarioTreeStateConstIterator GetNextStatesBegin() const;
	ScenarioTreeStateConstIterator GetNextStatesEnd() const;
	unsigned int GetNextStatesCount() const;

	friend class ScenarioTree;

protected:
	SCENINDEX index_; //index of the node starting from 0
	const ScenarioTree *tree_; //internal pointer to the tree
};

class TreeState {
public:
	TreeState(const ScenarioTree *tree, SCENINDEX parent_index, unsigned int state);
	TreeState();
	~TreeState();

	double GetProbability() const; //returns the probability of transition to this state
	unsigned int GetStage() const; //returns the stage of this state
	unsigned int GetState() const; //returns the state
	unsigned int GetDistributionIndex() const; //returns the indexing position of real distribution

	friend class ScenarioTree;

protected:
	const ScenarioTree *tree_; //internal pointer to the tree
	SCENINDEX parent_index_; //index of the parent node
	unsigned int state_; //state in current stage
};


/* stages are indexed 1 .. T , stages are also 1 .. N
*/
class ScenarioTree
{
public:
	ScenarioTree(unsigned int stages, const vector<unsigned int> &stage_samples, StageDependence dependence,
		const vector<Distribution*> &stage_distributions, boost::function<void(double *)> evaluate);
	ScenarioTree(unsigned int stages, vector<unsigned int> state_counts, 
		const vector<vector<unsigned int> > &stage_samples, StageDependence dependence,
		const vector<vector<Distribution*> > &stage_distributions, vector<mat> transition_probabilities, 
		boost::function<void(double *)> evaluate);
	~ScenarioTree(void);
	void GenerateTree();
	SCENINDEX ScenarioCount() const;
	unsigned int GetNodeSize(unsigned int stage) const;
	unsigned int StageCount() const;
	SCENINDEX ScenarioCountStage(unsigned int stage) const;
	SCENINDEX ScenarioCountState(unsigned int stage, unsigned int state) const;
	unsigned int DescendantCountStage(unsigned int stage) const;
	unsigned int DescendantCountTotalStage(unsigned int stage) const;
	unsigned int DescendantCountStage(unsigned int stage, unsigned int state) const;
	unsigned int StateCountStage(unsigned int stage) const;
	//counts nr. of scenarios to the given stage
	SCENINDEX ScenarioCountToStage(unsigned int to_stage) const;
	SCENINDEX ScenarioCountToState(unsigned int to_stage, unsigned int to_state) const;
	unsigned int StateCount(unsigned int to_stage) const;

	TreeNode GetRoot() const;
	TreeNode operator() (SCENINDEX index) const;
	TreeNode operator() (unsigned int stage, SCENINDEX index) const;
	TreeNode operator() (unsigned int stage, unsigned int state, SCENINDEX index) const;
	unsigned int GetStage(SCENINDEX index) const;
	unsigned int GetState(SCENINDEX index) const;
	const double * GetValues(SCENINDEX index) const;
	double GetProbability(SCENINDEX index) const;
	unsigned int GetDescendantsCount(SCENINDEX index) const;
	TreeNode GetParent(SCENINDEX index) const;
	ScenarioTreeConstIterator GetDescendantsBegin(SCENINDEX index) const;
	ScenarioTreeConstIterator GetDescendantsEnd(SCENINDEX index) const;
	bool HasParent(SCENINDEX index) const;
	unsigned int GetDescendantsCount(SCENINDEX index, unsigned int next_state) const;
	ScenarioTreeConstIterator GetDescendantsBegin(SCENINDEX index, unsigned int next_state) const;
	ScenarioTreeConstIterator GetDescendantsEnd(SCENINDEX index, unsigned int next_state) const;
	void ReduceSize(const vector<unsigned int> &stage_samples);
	void ReduceSize(const vector<vector<unsigned int> > &stage_samples);
	void ReduceOneScenario(unsigned int stage, unsigned int state);
	double ScenarioDistance(SCENINDEX s1, SCENINDEX s2) const;

	//indexing helpers for Markov chains
	unsigned int GetDistributionIndex(SCENINDEX index) const;
	unsigned int GetDistributionIndex(unsigned int stage, unsigned int state) const;
	ScenarioTreeStateConstIterator GetNextStatesBegin(SCENINDEX index) const;
	ScenarioTreeStateConstIterator GetNextStatesEnd(SCENINDEX index) const;
	double GetStateProbability(SCENINDEX index, unsigned int next_state) const;

protected:
	vector<vector<Distribution*> > stage_distributions_; //first index = stages, second = states
	boost::function<void(double *)> evaluate_;
	unsigned int stages_;
	//first index = stages, second = states
	vector<vector<unsigned int> > stage_samples_; //scenarios coming from each node at each stage (tree is not balanced)
	vector<unsigned int> state_counts_; //count of states for each stage
	StageDependence dependence_; //stage dependence of the distribution
	vector<mat> transition_probabilities_; //transition probabilities between states for each stage

	//samples to help independent sampling
	vector<mat> samples_;

	//samples to help independent sampling
	vector<colvec> probabilities_;

	//number of elements in each node, by stages
	vector<unsigned int> stage_node_size_;
	
	//destroys the whole scenario tree
	void DestroyTree();

	//real total of scenarios hold in mem
	unsigned int real_count_;

	//internal constructor
	void Init(unsigned int stages, vector<unsigned int> state_counts, 
		const vector<vector<unsigned int> > &stage_samples, StageDependence dependence,
		const vector<vector<Distribution*> > &stage_distributions, vector<mat> transition_probabilities,
		boost::function<void(double *)> evaluate);

	//internal setter
	void SetProbability(SCENINDEX index, double probability);

	//internal indexer
	unsigned int GetSampleIndex(SCENINDEX index) const;
};

class ScenarioTreeConstIterator : public boost::iterator_facade<
	ScenarioTreeConstIterator
	, TreeNode const
	, boost::bidirectional_traversal_tag //TODO: we can do more
	, TreeNode
> {
public:
	ScenarioTreeConstIterator() {
		tree_ = 0;
		index_ = 0;
	}

	explicit ScenarioTreeConstIterator(const ScenarioTree *tree, SCENINDEX index) {
		tree_ = tree;
		index_ = index;
	}

private:
	friend class boost::iterator_core_access;

	void increment() {
		++index_;
	}

	void decrement() {
		--index_;
	}

	bool equal(ScenarioTreeConstIterator const& other) const
	{
		return index_ == other.index_;
	}

	TreeNode const dereference() const {
		return (*tree_)(index_);
	}

	const ScenarioTree * tree_;
	SCENINDEX index_;
};

class ScenarioTreeStateConstIterator : public boost::iterator_facade<
	ScenarioTreeStateConstIterator
	, TreeState const
	, boost::bidirectional_traversal_tag //TODO: we can do more
	, TreeState
> {
public:
	ScenarioTreeStateConstIterator() {
		tree_ = 0;
		parent_index_ = 0;
		state_ = 0;
	}

	explicit ScenarioTreeStateConstIterator(const ScenarioTree *tree, SCENINDEX parent_index, unsigned int state) {
		tree_ = tree;
		parent_index_ = parent_index;
		state_ = state;
	}

private:
	friend class boost::iterator_core_access;

	void increment() {
		++state_;
	}

	void decrement() {
		--state_;
	}

	bool equal(ScenarioTreeStateConstIterator const& other) const
	{
		return (parent_index_ == other.parent_index_) && (state_ == other.state_);
	}

	TreeState const dereference() const {
		return TreeState(tree_, parent_index_, state_);
	}

	const ScenarioTree * tree_;
	SCENINDEX parent_index_;
	unsigned int state_;
};
