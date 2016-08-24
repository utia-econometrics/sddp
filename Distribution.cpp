#include "Distribution.h"

Distribution::Distribution(const colvec &mu, const mat &sigma)
{
	if(mu.n_rows != sigma.n_rows || mu.n_rows != sigma.n_cols) {
		throw DistributionException("Invalid distribution parameters");
	}
	if(min(sigma.diag()) < 0) {
		throw DistributionException("Invalid sigma.");
	}
	mu_ = mu;
	sigma_ = sigma;
	generator_ = RandomGenerator::GetGenerator();
}

Distribution::Distribution(const mat &sample)
{
	sample_ = sample;
	generator_ = RandomGenerator::GetGenerator();
}

Distribution::~Distribution(void)
{
	if(generator_ != 0) {
		delete generator_;
	}
}

void Distribution::SampleMean(rowvec &s_mean) {
	s_mean = mean(sample_, 0);
}

void Distribution::SampleStdDev(rowvec & s_stddev) {
	s_stddev = stddev(sample_, 1, 0);
}

void Distribution::SampleSkewness(rowvec & s_skewness) {
	rowvec s_mean = mean(sample_, 0);
	rowvec s_stddev = stddev(sample_, 1, 0);
	int dimension = s_mean.n_cols;
	int count = sample_.n_rows;
	rowvec tmp = zeros(1, dimension);
	for(int i = 0; i < count; ++i) {
		tmp += pow(sample_.row(i) - s_mean, 3);
	}
	rowvec scale(dimension);
	scale.fill(1.0 / count);
	s_skewness = tmp % scale % pow(s_stddev, -3);
}

void Distribution::SampleKurtosis(rowvec & s_kurtosis) {
	rowvec s_mean = mean(sample_, 0);
	rowvec s_stddev = stddev(sample_, 1, 0);
	int dimension = s_mean.n_cols;
	int count = sample_.n_rows;
	rowvec tmp = zeros(1, dimension);
	for(int i = 0; i < count; ++i) {
		tmp += pow(sample_.row(i) - s_mean, 4);
	}
	rowvec scale(dimension);
	scale.fill(1.0 / count);
	s_kurtosis = tmp % scale % pow(s_stddev, -4);
}

void Distribution::UnbiasedEstimate() {
	mu_ = trans(mean(sample_, 0));
	sigma_ = this->GetAlpha() * cov(sample_, sample_, 0);

	//do not modify with alpha, for moments
	corr_ = cor(sample_, sample_, 0);
}

double Distribution::GetAlpha() const {
	return 1;
}


void Distribution::MLEstimate() {
	mu_ = trans(mean(sample_, 0));
	//not using alpha, need to override!
	sigma_ = cov(sample_, sample_, 1);
	//never use alpha, use unbiased version, should be the same as ML, hence armadillo gives wrong results
	corr_ = cor(sample_, sample_, 0);
}