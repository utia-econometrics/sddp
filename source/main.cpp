#include <armadillo>
#include "NormalDistribution.h"
#include "ScenarioTree.h"
#include "SaaSolver.h"
#include "SaaNestedSolver.h"
#include "SddpSolver.h"
#include "AnalyticalSolver.h"
#include "GeometricBrownianMotion.h"
#include "CorrelatedNormal.h"
#include <boost/filesystem/fstream.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/lexical_cast.hpp>
#include "VarianceEstimator.h"

using namespace std;
using namespace arma;
using namespace boost;

const unsigned int STAGES = 10;
const unsigned int ITERATIONS = 10;
const unsigned int SEED = 350916502;
const unsigned int DESCENDANTS = 1000;
const unsigned int REDUCED_DESCENDANTS = 0; //0 == no reduction
const unsigned int DERIVATIVE_ITERATIONS = 10;
const double TRANSACTION_COSTS = 0.000; //0.003

#define ANALYZE_CONTAMINATION 0
#define COMPUTE_CONTAMINATION_BOUND 0
#define DO_VARIANCE 0


void variance() {
	int iteration = 0;
	VarianceEstimator estim;
	filesystem::ofstream mean_file;
	filesystem::ofstream variance_file;
	mean_file.open("mean.txt", ios_base::out);
	variance_file.open("variance.txt", ios_base::out);
	double alpha = 0.05;
	mean_file << "lambda/prob";
	variance_file << "lambda/prob";
	for(double lambda = 0.01; lambda < 1.00; lambda += 0.01) {
		mean_file << "\t" << lambda;
		variance_file << "\t" << lambda;
	}
	mean_file << endl;
	variance_file << endl;
	for(double prob = 0.01; prob < 1.00; prob += 0.01) {
		mean_file << prob;
		variance_file << prob;
		for(double lambda = 0.01; lambda < 1.00; lambda += 0.01) {
			cout << "Iteration " << iteration++ << endl;
			double mean;
			double variance;
			estim.EstimateNormal(alpha, lambda, prob, mean, variance);
			mean_file << "\t" << mean;
			variance_file << "\t" << variance;
		}
		mean_file << endl;
		variance_file << endl;	
	}
	mean_file.close();
	variance_file.close();
}

void varianceDiscrete(string append = "") {
	//load the data matrix
	filesystem::ifstream datafile;
	string line;
	string in_file_name = "scenarios";
	string mean_file_name = "mean";
	string variance_file_name = "variance";
	if(append != "") {
		in_file_name += append;	
		mean_file_name += append;
		variance_file_name += append;
	}
	in_file_name += ".txt";	
	mean_file_name += ".txt";
	variance_file_name += ".txt";
	datafile.open(in_file_name);
	stringstream str;
	while(getline(datafile,line)) {
		str << line << endl;
	}
	datafile.close();
	mat data;
	data.load(str);
	colvec scenarios = data.col(0);

	int iteration = 0;
	VarianceEstimator estim;
	filesystem::ofstream mean_file;
	filesystem::ofstream variance_file;
	mean_file.open(mean_file_name, ios_base::out);
	variance_file.open(variance_file_name, ios_base::out);
	double alpha = 0.05;
	mean_file << "lambda/prob";
	variance_file << "lambda/prob";
	//for(double lambda = 0.01; lambda < 1.00; lambda += 0.01) {
	for(double lambda = 0.1; lambda < 1.00; lambda += 0.10) {
		mean_file << "\t" << lambda;
		variance_file << "\t" << lambda;
	}
	mean_file << endl;
	variance_file << endl;
	for(double prob = 0.01; prob < 1.00; prob += 0.01) {
		mean_file << prob;
		variance_file << prob;
		//for(double lambda = 0.01; lambda < 1.00; lambda += 0.01) {
		for(double lambda = 0.1; lambda < 1.00; lambda += 0.10) {
			cout << "Iteration " << iteration++ << endl;
			double mean;
			double variance;
			estim.EstimateDiscrete(scenarios, alpha, lambda, prob, mean, variance);
			mean_file << "\t" << mean;
			variance_file << "\t" << variance;
		}
		mean_file << endl;
		variance_file << endl;	
	}
	mean_file.close();
	variance_file.close();
}

int main(int argc, char *argv[], char *envp[]) {

#if DO_VARIANCE == 1
	//variance();
	
	for(int i = 2; i <= 15; ++i) {
		stringstream str_app;
		str_app << i;
		varianceDiscrete(str_app.str());
	}
	
	int aaa;
	cin >> aaa;
	return 0;
#endif

	if(argc < 2) {
		cout << "No input data file provided." << endl;
		return -1;
	}

	filesystem::ifstream datafile;
	datafile.open(argv[1]);
	if(!datafile.is_open()) {
		cout << "Data file loading failed." << endl;
		return -2;
	}

	string line;
	if(!getline(datafile,line)) {
		cout << "No header supplied in data file." << endl;
		return -3;
	}

	//parse asset names into the vector
	vector<string> assets;
	split(assets, line, is_any_of(" \t"));

	if(assets.empty()) {
		cout << "Invalid header in the data file." << endl;
		return -4;
	}

	//load the data matrix
	stringstream str;
	while(getline(datafile,line)) {
		str << line << endl;
	}
	datafile.close();
	mat data;
	data.load(str);

	if(data.n_cols != assets.size()) {
		cout << "Header line does not match data." << endl;
		return -5;
	}
	if(data.n_rows == 0) {
		cout << "No data supplied." << endl;
		return -6;
	}

	RiskParameters param;
	param.confidence = 0.95;
	param.confidence_other = 0.99;
//	param.risk_measure = RISK_CVAR_MULTIPERIOD;
	param.risk_measure = RISK_CVAR_NESTED;
	param.discount_factor = 1.00;
	param.transaction_costs = TRANSACTION_COSTS;
	//first coefs are zero
	param.risk_coefficients.push_back(0.0);
	param.risk_coefficients_other.push_back(0.0);
	param.expectation_coefficients.push_back(0.0);
	for(unsigned int i = 1; i < STAGES; ++i) {
//		double r_coef = i / static_cast<double>(STAGES);
		double r_coef = 0.5;
//		double r_coef = 0.2;
//		double r_coef = 0.1; //meaningful according to the Brazilian study for alpha=5%
		param.risk_coefficients.push_back(r_coef);
		param.risk_coefficients_other.push_back(0.0);
//		param.risk_coefficients.push_back(r_coef / 2);
//		param.risk_coefficients_other.push_back(r_coef / 2);
		param.expectation_coefficients.push_back(1 - r_coef);
	}

	vector<rowvec> vec_weights_o;
	vector<double> vec_lb_o;
	vector<double> vec_ub_o_m;
	vector<double> vec_ub_o_b;
	vector<time_t> vec_time_o;
	
	vector<rowvec> vec_weights_c;
	vector<double> vec_lb_c;
	vector<double> vec_ub_c_m;
	vector<double> vec_ub_c_b;
	vector<time_t> vec_time_c;

	vector<double> vec_der_c_mean;
	vector<double> vec_der_c_bound;
	vector<time_t> vec_time_der;

	filesystem::ofstream output_file;
	output_file.open("output.txt", ios_base::in);

	for(unsigned int i = 0; i < ITERATIONS; ++i) {
	
		output_file << "Iteration " << i << ":" << endl;

		//set the seed
		if(SEED > 0) {
			RandomGenerator::SetSeed(SEED + i); //fixes the tree for each iteration
		}

		rowvec weights_o;
		double lb_o;
		double ub_o_m;
		double ub_o_b;
		time_t time_o;

		time_o = time(NULL);

		GeometricBrownianMotion model(STAGES, assets, param, data);
		SddpSolver solver(&model, DESCENDANTS, REDUCED_DESCENDANTS);
	
		solver.Solve(weights_o, lb_o, ub_o_m, ub_o_b);
		time_o = time(NULL) - time_o;

		vec_weights_o.push_back(weights_o);
		vec_lb_o.push_back(lb_o);
		vec_ub_o_m.push_back(ub_o_m);
		vec_ub_o_b.push_back(ub_o_b);
		vec_time_o.push_back(time_o);

#if ANALYZE_CONTAMINATION == 1
		//contamination
		rowvec weights_c;
		double lb_c;
		double ub_c_m;
		double ub_c_b;
		time_t time_c;

		time_c = time(NULL);
		colvec mu = model.GetMu();
		mat sigma = model.GetSigma();
		sigma *= 1.2; //20% more variance
		GeometricBrownianMotion model_cont(STAGES, assets, param, mu, sigma);
		SddpSolver solver_cont(&model_cont, DESCENDANTS);
	
		solver_cont.Solve(weights_c, lb_c, ub_c_m, ub_c_b);
		time_c = time(NULL) - time_c;

		vec_weights_c.push_back(weights_c);
		vec_lb_c.push_back(lb_c);
		vec_ub_c_m.push_back(ub_c_m);
		vec_ub_c_b.push_back(ub_c_b);
		vec_time_c.push_back(time_c);

#if COMPUTE_CONTAMINATION_BOUND == 1
		//contamination bounds
		double der_c_mean;
		double der_c_bound;
		time_t time_der;
		time_der = time(NULL);
		boost::function<vector<vector<double>>(vector<const double *>)> policy = boost::bind(&SddpSolver::GetPolicy, &solver, _1);
		solver_cont.EvaluatePolicy(policy, der_c_mean, der_c_bound, DERIVATIVE_ITERATIONS);
		time_der = time(NULL) - time_der;

		vec_der_c_mean.push_back(der_c_mean);
		vec_der_c_bound.push_back(der_c_bound);
		vec_time_der.push_back(time_der);
#endif
#endif

		//printout
		output_file << "Original problem: " << weights_o << endl;
#if ANALYZE_CONTAMINATION == 1
		output_file << "Perturbed problem: " << weights_c << endl;
#endif
		output_file << "Original problem @ " << time_o << "s, lb: " << lb_o << ", ub_mean:" << ub_o_m << ", ub_bound:" << ub_o_b << endl ;
#if ANALYZE_CONTAMINATION == 1
		output_file << "Perturbed problem @ " << time_c << "s, lb: " << lb_c << ", ub_mean:" << ub_c_m << ", ub_bound:" << ub_c_b << endl ;
#if COMPUTE_CONTAMINATION_BOUND == 1
		output_file << "Contamination bound @ " << time_der << "s, mean:" << der_c_mean << ", bound:" << der_c_bound << endl ;
#endif
#endif

	}

	output_file.close();
	
	for(unsigned int i = 0; i < ITERATIONS; ++i) {
		cout << "Original problem: " << vec_weights_o[i] << endl;
	}
#if ANALYZE_CONTAMINATION == 1
	for(unsigned int i = 0; i < ITERATIONS; ++i) {
		cout << "Perturbed problem: " << vec_weights_c[i] << endl;
	}
#endif
	for(unsigned int i = 0; i < ITERATIONS; ++i) {
		cout << "Original problem @ " << vec_time_o[i] << "s, lb: " << vec_lb_o[i] << ", ub_mean:" << vec_ub_o_m[i] << ", ub_bound:" << vec_ub_o_b[i] << endl ;
	}
#if ANALYZE_CONTAMINATION == 1
	for(unsigned int i = 0; i < ITERATIONS; ++i) {
		cout << "Perturbed problem @ " << vec_time_c[i] << "s, lb: " << vec_lb_c[i] << ", ub_mean:" << vec_ub_c_m[i] << ", ub_bound:" << vec_ub_c_b[i] << endl ;
	}
#endif
#if ANALYZE_CONTAMINATION == 1
#if COMPUTE_CONTAMINATION_BOUND == 1
	for(unsigned int i = 0; i < ITERATIONS; ++i) {
		cout << "Contamination bound @ " << vec_time_der[i] << "s, mean:" << vec_der_c_mean[i] << ", bound:" << vec_der_c_bound[i] << endl ;
	}
#endif
#endif

	//even when debugging do not end
	char c;
	cin >> c;

	return 0;
}
