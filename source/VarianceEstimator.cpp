#include "VarianceEstimator.h"
#include "NormalDistribution.h"
#include "LognormalDistribution.h"
#include "DiscreteDistribution.h"

#define RUN_ITERATIONS 10000000 //10000

VarianceEstimator::VarianceEstimator(void)
{
		generator_ = RandomGenerator::GetGenerator();
}


VarianceEstimator::~VarianceEstimator(void)
{
}

void VarianceEstimator::EstimateLognormal(double alpha, double lambda, double probability, double &mean, double &variance) {
	unsigned int counter = 0;
	double* results = new double[RUN_ITERATIONS];
	double* probabilities = new double[RUN_ITERATIONS];
	colvec mu = zeros(1);
	mat sigma = ones(1,1);
	
/* OPRAS */
	mu(0) = 0;
	sigma(0,0) = 1;
/* KONEC OPRASU*/

	LognormalDistribution distribution(mu, sigma);
	double margin = distribution.InverseDistributionFunction(1-alpha);
	for(int counter = 0; counter < RUN_ITERATIONS; ++counter) {
		//coin flip
		double sample;
		mat sample_mat;
		double weight;
		double value = lambda * margin;
		if(generator_->GetRandom() < probability) {
			//conditional node
			do {
				distribution.GenerateAnalytic(sample_mat, 1);
				sample = sample_mat(0, 0);
			} while(sample < margin);
			value += (lambda / alpha) * (sample - margin);
			weight = (1.0 / probability) * alpha;
		}
		else {	
			do {
				distribution.GenerateAnalytic(sample_mat, 1);
				sample = sample_mat(0, 0);
			} while(sample >= margin);
			weight = (1.0 / (1 - probability)) * (1 - alpha);
		}
		value += (1 - lambda) * sample;
		value *= weight;
		results[counter] = value;
		probabilities[counter] = weight;
	}

	double probability_sum = 0;
	for(int counter = 0; counter < RUN_ITERATIONS; ++counter) {
		probability_sum += probabilities[counter];
	}

	double sum = 0;
	for(int counter = 0; counter < RUN_ITERATIONS; ++counter) {
		sum += results[counter];
	}
	//probabilities sum up to one
	mean = sum / probability_sum;
	sum = 0;
	for(int counter = 0; counter < RUN_ITERATIONS; ++counter) {
		sum += pow(results[counter] - mean, 2.0);
	}
	variance = sum / probability_sum;
	delete results;
	delete probabilities;
}

void VarianceEstimator::EstimateNormal(double alpha, double lambda, double probability, double &mean, double &variance) {
	unsigned int counter = 0;
	double* results = new double[RUN_ITERATIONS];
	double* probabilities = new double[RUN_ITERATIONS];
	colvec mu = ones(1) * 10;
	mat sigma = ones(1,1) * 2;
	NormalDistribution distribution(mu, sigma);
	double margin = distribution.InverseDistributionFunction(1-alpha);
	for(int counter = 0; counter < RUN_ITERATIONS; ++counter) {
		//coin flip
		double sample;
		mat sample_mat;
		double weight;
		double value = lambda * margin;
		if(generator_->GetRandom() < probability) {
			//conditional node
			do {
				distribution.GenerateAnalytic(sample_mat, 1);
				sample = sample_mat(0, 0);
			} while(sample < margin);
			value += (lambda / alpha) * (sample - margin);
			weight = (1.0 / probability) * alpha;
		}
		else {	
			do {
				distribution.GenerateAnalytic(sample_mat, 1);
				sample = sample_mat(0, 0);
			} while(sample >= margin);
			weight = (1.0 / (1 - probability)) * (1 - alpha);
		}
		value += (1 - lambda) * sample;
		value *= weight;
		results[counter] = value;
		probabilities[counter] = weight;
	}

	double probability_sum = 0;
	for(int counter = 0; counter < RUN_ITERATIONS; ++counter) {
		probability_sum += probabilities[counter];
	}

	double sum = 0;
	for(int counter = 0; counter < RUN_ITERATIONS; ++counter) {
		sum += results[counter];
	}
	//probabilities sum up to one
	mean = sum / probability_sum;
	sum = 0;
	for(int counter = 0; counter < RUN_ITERATIONS; ++counter) {
		sum += pow(results[counter] - mean, 2.0);
	}
	variance = sum / probability_sum;
	delete results;
	delete probabilities;
}

void VarianceEstimator::EstimateDiscrete(colvec &scenarios, double alpha, double lambda, double probability, double &mean, double &variance) {
	unsigned int counter = 0;
	double* results = new double[RUN_ITERATIONS];
	double* probabilities = new double[RUN_ITERATIONS];
	DiscreteDistribution distribution(scenarios);
	double margin = distribution.InverseDistributionFunction(1-alpha);
	for(int counter = 0; counter < RUN_ITERATIONS; ++counter) {
		//coin flip
		double sample;
		mat sample_mat;
		double weight;
		double value = lambda * margin;
		if(generator_->GetRandom() < probability) {
			//conditional node
			do {
				distribution.GenerateAnalytic(sample_mat, 1);
				sample = sample_mat(0, 0);
			} while(sample < margin);
			value += (lambda / alpha) * (sample - margin);
			weight = (1.0 / probability) * alpha;
		}
		else {	
			do {
				distribution.GenerateAnalytic(sample_mat, 1);
				sample = sample_mat(0, 0);
			} while(sample >= margin);
			weight = (1.0 / (1 - probability)) * (1 - alpha);
		}
		value += (1 - lambda) * sample;
		value *= weight;
		results[counter] = value;
		probabilities[counter] = weight;
	}

	double probability_sum = 0;
	for(int counter = 0; counter < RUN_ITERATIONS; ++counter) {
		probability_sum += probabilities[counter];
	}

	double sum = 0;
	for(int counter = 0; counter < RUN_ITERATIONS; ++counter) {
		sum += results[counter];
	}
	//probabilities sum up to one
	mean = sum / probability_sum;
	sum = 0;
	for(int counter = 0; counter < RUN_ITERATIONS; ++counter) {
		sum += pow(results[counter] - mean, 2.0);
	}
	variance = sum / probability_sum;
	delete results;
	delete probabilities;
}
