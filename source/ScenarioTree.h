#pragma once

#include "Distribution.h"
#include "Exception.h"
#include <boost/function.hpp>
#include "Helper.h"
#include <boost/iterator/iterator_facade.hpp>

using namespace std;

class ScenarioTree; //forward
class ScenarioTreeConstIterator; //forward

//runtime exception
class ScenarioTreeException : Exception
{ 
public:
	ScenarioTreeException(string text) : Exception(text){
	}
};

enum StageDependence {
	STAGE_INDEPENDENT = 0,
	/* to be implemented */
};

class TreeNode {
public:
	TreeNode(const ScenarioTree *tree, SCENINDEX index);
	TreeNode();
	~TreeNode();

	SCENINDEX GetNumber() const; //nubmer of the scenario
	unsigned int GetStage() const; //stage of the scenario
	const double * GetValues() const; //prices of assets
	double GetProbability() const; //probability of this scenario
	ScenarioTreeConstIterator GetDescendantsBegin() const; //following stages
	ScenarioTreeConstIterator GetDescendantsEnd() const; //following stages
	unsigned int GetDescendantsCount() const; //scenarios number in next stage
	TreeNode GetParent() const; //pointer to the parent
	bool HasParent() const; //true if has parent
	SCENINDEX GetIndex() const; //true index
	SCENINDEX GetStageIndex() const; //index with respect to the node's stage
	unsigned int GetSize() const; //number of values in the node

	friend class ScenarioTree;

protected:
	SCENINDEX index_; //index of the node starting from 0
	const ScenarioTree *tree_; //internal pointer to the tree
};

class ScenarioTree
{
public:
	ScenarioTree(unsigned int stages, const vector<unsigned int> &stage_samples, StageDependence dependence,
		const vector<Distribution*> &stage_distributions, boost::function<void(double *)> evaluate);
	~ScenarioTree(void);
	void GenerateTree();
	SCENINDEX ScenarioCount() const;
	unsigned int GetNodeSize(unsigned int stage) const;
	unsigned int StageCount() const;
	SCENINDEX ScenarioCountStage(unsigned int stage) const;
	unsigned int DescendantCountStage(unsigned int stage) const;
	//counts nr. of scenarios to the given stage
	SCENINDEX ScenarioCount(unsigned int to_stage) const;
	SCENINDEX ScenarioSize(unsigned int to_stage) const;

	TreeNode GetRoot() const;
	TreeNode operator() (SCENINDEX index) const;
	TreeNode operator() (unsigned int stage, SCENINDEX index) const;
	unsigned int GetStage(SCENINDEX index) const;
	double * GetValues(SCENINDEX index) const;
	double GetProbability(SCENINDEX index) const;
	unsigned int GetDescendantsCount(SCENINDEX index) const;
	SCENINDEX GetStageIndex(SCENINDEX index) const;
	TreeNode GetParent(SCENINDEX index) const;
	ScenarioTreeConstIterator GetDescendantsBegin(SCENINDEX index) const;
	ScenarioTreeConstIterator GetDescendantsEnd(SCENINDEX index) const;
	ScenarioTreeConstIterator GetStageBegin(unsigned int stage) const;
	ScenarioTreeConstIterator GetStageEnd(unsigned int stage) const;
	bool HasParent(SCENINDEX index) const;
	void ReduceSize(const vector<unsigned int> &stage_samples);
	void ReduceOneScenario(unsigned int stage);
	double ScenarioDistance(SCENINDEX s1, SCENINDEX s2) const;

protected:
	vector<Distribution*> stage_distributions_;
	boost::function<void(double *)> evaluate_;
	unsigned int stages_;
	vector<unsigned int> stage_samples_; //scenarios coming from each node at each stage (tree is not balanced)
	StageDependence dependence_; //stage dependence of the distribution

	//samples to help independent sampling
	mat *samples_;

	//samples to help independent sampling
	colvec *probabilities_;

	//number of elements in each node, by stages
	vector<unsigned int> stage_node_size_;
	
	//destroys the whole scenario tree
	void DestroyTree();

	//real total of scenarios hold in mem
	unsigned int real_count_;

	//internal genetators
	void GenerateTreeIndependent(void);

	//internal setter
	void SetProbability(SCENINDEX index, double probability);
};

class ScenarioTreeConstIterator : public boost::iterator_facade<
	ScenarioTreeConstIterator
	, TreeNode const
	, boost::bidirectional_traversal_tag //TODO: we can do more
	, TreeNode
>{
public:
	ScenarioTreeConstIterator() {
		tree_ = 0;
		index_ = 0;
	}

	explicit ScenarioTreeConstIterator(const ScenarioTree *tree, SCENINDEX index)  {
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
