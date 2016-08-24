#include "DiscreteDistribution.h"

using namespace arma;
using namespace std;

DiscreteDistribution::DiscreteDistribution(const mat &sample) : Distribution(sample)
{

}

DiscreteDistribution::~DiscreteDistribution(void)
{

}

void DiscreteDistribution::GenerateAnalytic(mat &sample, int count)
{
	//TODO: scenario generating according to 4 moments??
	int dimension = sample_.n_cols;
	sample.set_size(count, dimension);
	int size = sample_.n_rows;

	for(unsigned int i = 0; i < count; ++i) {
		int index = generator_->GetRandomInt(0, size - 1);
		sample.row(i) = sample_.row(index);
	}
}

double DiscreteDistribution::DistributionFunction(double x) const {
	int dimension = sample_.n_cols;
	int size = sample_.n_rows;
	if(dimension > 1) {
		throw DistributionException("Cannot calculate inverse distribution function for multivariate random variable");
	}
	colvec sorted = sample_.col(0);
	sort(sorted);
	unsigned int i;
	for(i = 0; i < size; ++i) {
		if(sorted(i) > x) {
			break;
		}
	}
	return i / size;
}

double DiscreteDistribution::InverseDistributionFunction(double y) const {
	int dimension = sample_.n_cols;
	int size = sample_.n_rows;
	if(dimension > 1) {
		throw DistributionException("Cannot calculate inverse distribution function for multivariate random variable");
	}
	colvec scenarios = sample_.col(0);
	colvec sorted = sort(scenarios);
	int position = ceil(y * size); //position in number of elements 1 .. size
	return sorted(--position); //0-based indices
}

DistributionType DiscreteDistribution::GetType() const {
	return DISTRIBUTION_DISCRETE;
}