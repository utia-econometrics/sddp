#include "NormalDistribution.h"
#include "alglib/specialfunctions.h"

using namespace alglib;
using namespace arma;

NormalDistribution::NormalDistribution(const colvec &mu, const mat &sigma) : Distribution(mu, sigma)
{
	spare_sample_ready_ = false;
}

NormalDistribution::NormalDistribution(const mat &sample) : Distribution(sample)
{
	spare_sample_ready_ = false;
}

NormalDistribution::~NormalDistribution(void)
{

}

void NormalDistribution::GenerateAnalytic(mat &sample, int count)
{
	int dimension = mu_.n_elem;

	//get the sigma decomposition
	colvec eigval;
	mat eigvec;
	mat lambda(dimension, dimension);
	eig_sym(eigval, eigvec, sigma_);
	lambda.fill(0);
	lambda.diag() = eigval;

	//sample the standardized N(0,1)
	mat standardized(count, dimension);
	for(int col = 0; col < dimension; ++col) {
		for(int row = 0; row < count; ++row) {
			standardized(row, col) = GenerateStandardized();
		}
	}

	//transform the data and return
	mat mu_add(count, dimension);
	mu_add = ones<colvec>(count) * trans(mu_);
	sample = mu_add + standardized * sqrt(lambda) * trans(eigvec);
}

double NormalDistribution::GenerateStandardized() {
	//Marsaglia polar method
	if(spare_sample_ready_) {
		spare_sample_ready_ = false;
		return spare_sample_;
	}
	double u, v, s;
	do {
		u = generator_->GetRandom() * 2 - 1;
		v = generator_->GetRandom() * 2 - 1;
		s = u * u + v * v;
	} while (s >= 1 || s <= 0);
	spare_sample_ = v * sqrt(-2 * log(s) / s);
	spare_sample_ready_ = true;
	return u * sqrt(-2 * log(s) / s);
}

double NormalDistribution::DistributionFunction(double x) const {
	if(mu_.n_rows != 1) {
		throw DistributionException("This function is for univariate normal distribution only.");
	}
	double mu = mu_(0);
	double sigma = sqrt(sigma_(0, 0));
	return (1 / sigma) * normaldistribution( (x - mu) / sigma);
}

double NormalDistribution::InverseDistributionFunction(double y) const {
	if(mu_.n_rows != 1) {
		throw DistributionException("This function is for univariate normal distribution only.");
	}
	double mu = mu_(0);
	double sigma = sqrt(sigma_(0, 0));
	return mu + sigma * invnormaldistribution(y);
}

double NormalDistribution::SampleJarqueBera() {
	if(sample_.n_cols != 1) {
		throw DistributionException("This test is for univariate normal distribution only.");
	}
	rowvec kurt;
	rowvec skew;
	SampleKurtosis(kurt);
	SampleSkewness(skew);
	double jb = (sample_.n_rows / 6) * (pow(skew(0), 2) + pow(kurt(0) - 3, 2) / 4);
	return 1 - chisquaredistribution(2, jb); //p-value
}

DistributionType NormalDistribution::GetType() const {
	return DISTRIBUTION_NORMAL;
}
