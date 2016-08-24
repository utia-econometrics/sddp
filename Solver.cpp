#include "Solver.h"

Solver::Solver(const ScenarioModel *model) {
	model_ = model;
	tree_ = 0;
	tree_forced_ = false;
}

Solver::~Solver(void) {
	this->Clean();
}

ScenarioTree* Solver::GetTree(bool regenerate) {
	if((regenerate || (tree_ == 0)) && !tree_forced_) {
		//clean up
		if(tree_ != 0) {
			delete tree_;
		}
		//make new tree
		vector<unsigned int> stage_samples;
		this->GetStageSamples(stage_samples);
		tree_ = model_->GetScenarioTree(stage_samples);

		//apply scenario reduction (if any)
		vector<unsigned int> reduced_samples;
		this->GetReducedSamples(reduced_samples);
		tree_->ReduceSize(reduced_samples);
	}
	return tree_;
}

void Solver::ForceTree(ScenarioTree *tree) {
	this->Clean();
	tree_ = tree;
	tree_forced_ = true;
}

void Solver::Clean() {
	if((tree_ != 0) && !tree_forced_) {
		delete tree_;
	}
}

void Solver::GetReducedSamples(vector<unsigned int> &stage_samples) {
	//no reduction
	return this->GetStageSamples(stage_samples); 
}
