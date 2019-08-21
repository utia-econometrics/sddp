#include <armadillo>
#include "NormalDistribution.h"
#include "ScenarioTree.h"
#include "SddpSolver.h"
#include "AssetAllocationModel.h"
#include <boost/filesystem/fstream.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/lexical_cast.hpp>
#include "VarianceEstimator.h"

using namespace std;
using namespace arma;
using namespace boost;

const unsigned int STAGES = 3;
const unsigned int ITERATIONS = 1;
const unsigned int SEED = 350916502;
const unsigned int DESCENDANTS = 10; //100
const unsigned int REDUCED_DESCENDANTS = 0; //0 == no reduction
const unsigned int DERIVATIVE_ITERATIONS = 10;

//runs SddpSolver for AssetAllocationModel
AssetAllocationModel* assetModel(string inputFile) {
	filesystem::ifstream datafile;
	datafile.open(inputFile);
	if (!datafile.is_open()) {
		throw Exception("Data file loading failed.");
	}

	string line;
	if (!getline(datafile, line)) {
		throw Exception("No header supplied in data file.");
	}

	//parse asset names into the vector
	vector<string> assets;
	split(assets, line, is_any_of(" \t"));

	if (assets.empty()) {
		throw Exception("Invalid header in the data file.");
	}

	//load the data matrix
	stringstream str;
	while (getline(datafile, line)) {
		str << line << endl;
	}
	datafile.close();
	mat data;
	data.load(str);

	if (data.n_cols != assets.size()) {
		throw Exception("Header line does not match data.");
	}
	if (data.n_rows == 0) {
		throw Exception("No data supplied.");
	}

	//Configure our AssetAllocationModel
	RiskParameters param;
	param.confidence = 0.95;
	param.confidence_other = 0.99;
	param.risk_measure = RISK_CVAR_NESTED; //or RISK_CVAR_MULTIPERIOD
	param.stage_dependence = MARKOV; // STAGE_INDEPENDENT
	param.markov_crisis_variance_factor = 1.3;
	param.markov_to_crisis_probability = 0.1;
	param.markov_from_crisis_probability = 0.6;
	param.discount_factor = 1.00;
	param.transaction_costs = 0.0; //0.003
								   //first coefs need to be zero
	param.risk_coefficients.push_back(0.0);
	param.risk_coefficients_other.push_back(0.0);
	param.expectation_coefficients.push_back(0.0);
	for (unsigned int i = 1; i < STAGES; ++i) {
		double r_coef = 0.1; //meaningful according to the Brazilian study for alpha=5%
		param.risk_coefficients.push_back(r_coef);
		param.risk_coefficients_other.push_back(0.0);
		param.expectation_coefficients.push_back(1 - r_coef);
	}

	return new AssetAllocationModel(STAGES, assets, param, data);
}

int main(int argc, char *argv[], char *envp[]) {
	if (argc < 2) {
		cout << "No model specified." << endl;
		return -1;
	}

	string problem = argv[1];

	string filename;
	if (argc >= 3) {
		filename = argv[2];
	}

	//set the seed
	if (SEED > 0) {
		RandomGenerator::SetSeed(SEED); //fixes the seed for model
	}
	
	ScenarioModel *model;
	try {
		if (problem == "asset") {
			model = assetModel(filename);
		}
		else {
			cout << "Invalid model specified." << endl;
			return -2;
		}
	}
	catch (Exception e) {
		cout << e.what() << endl;
		return -3;
	}

    //for contamination
    AssetAllocationModel *model_cont = assetModel(filename);
    model_cont->SetSigma(model_cont->GetSigma() * 1.2); //20% more variance

    //configure the SddpSolver
    SddpSolverConfig config;
    config.samples_per_stage = DESCENDANTS;
    config.reduced_samples_per_stage = REDUCED_DESCENDANTS;
    config.solver_strategy = STRATEGY_DEFAULT; // STRATEGY_CONDITIONAL
    // config.cut_nodes_not_tail = true;
    //there are more settings, for instace:
    //config.debug_solver = true;
    config.calculate_future_solutions = true;
    config.calculate_future_solutions_count = 1000;


	//vectors for collecting stats and printout
	vector<rowvec> vec_weights_o;
	vector<double> vec_lb_o;
	vector<double> vec_ub_o_m;
	vector<double> vec_ub_o_b;
	vector<time_t> vec_time_o;

    //contamination
    vector<rowvec> vec_weights_c;
    vector<double> vec_lb_c;
    vector<double> vec_ub_c_m;
    vector<double> vec_ub_c_b;
    vector<time_t> vec_time_c;

    vector<double> vec_der_c_mean;
    vector<double> vec_der_c_bound;
    vector<time_t> vec_time_der;

    rowvec weights_c;
    double lb_c;
    double ub_c_m;
    double ub_c_b;
    time_t time_c;

	filesystem::ofstream output_file;
	output_file.open("output.txt", ios_base::in);

	for (unsigned int i = 0; i < ITERATIONS; ++i) {

		output_file << "Iteration " << i << ":" << endl;

		//set the seed
		if (SEED > 0) {
			RandomGenerator::SetSeed(SEED + i + 1); //fixes the tree for each iteration
		}

		//vector for statistics and output
		rowvec weights_o;
		double lb_o;
		double ub_o_m;
		double ub_o_b;
		time_t time_o;
        vector<mat> fut_sol;

		time_o = time(NULL);

		SddpSolver solver(model, config);

        solver.Solve(weights_o, lb_o, ub_o_m, ub_o_b, fut_sol);
		time_o = time(NULL) - time_o;

		vec_weights_o.push_back(weights_o);
		vec_lb_o.push_back(lb_o);
		vec_ub_o_m.push_back(ub_o_m);
		vec_ub_o_b.push_back(ub_o_b);
		vec_time_o.push_back(time_o);

		//printout
		output_file << "Original problem: " << weights_o << endl;
		output_file << "Original problem @ " << time_o << "s, lb: " << lb_o << ", ub_mean:" << ub_o_m << ", ub_bound:" << ub_o_b << endl;

        for(unsigned int stage = 1; stage <= STAGES; ++stage) {
            cout << "Stage " << stage << " solution:" << endl;
            cout << mean(fut_sol[stage - 1]) << endl;
            cout << stddev(fut_sol[stage - 1]) << endl;
        }

        time_c = time(NULL);
	    SddpSolver solver_cont(model_cont, config);

        solver_cont.Solve(weights_c, lb_c, ub_c_m, ub_c_b);
        time_c = time(NULL) - time_c;

        vec_weights_c.push_back(weights_c);
        vec_lb_c.push_back(lb_c);
        vec_ub_c_m.push_back(ub_c_m);
        vec_ub_c_b.push_back(ub_c_b);
        vec_time_c.push_back(time_c);

        //contamination bounds
        double der_c_mean;
        double der_c_variance;
        double der_c_bound;
        time_t time_der;
        time_der = time(NULL);
        boost::function<vector<vector<double>>(vector<const double *>, vector<unsigned int>)> policy = boost::bind(&SddpSolver::GetPolicy, &solver, _1, _2);
        solver_cont.EvaluatePolicy(policy, der_c_mean, der_c_variance, der_c_bound, DERIVATIVE_ITERATIONS);
        time_der = time(NULL) - time_der;

        vec_der_c_mean.push_back(der_c_mean);
        vec_der_c_bound.push_back(der_c_bound);
        vec_time_der.push_back(time_der);
	}

	output_file.close();

	for (unsigned int i = 0; i < ITERATIONS; ++i) {
		cout << "Original problem: " << vec_weights_o[i] << endl;
	}
	for (unsigned int i = 0; i < ITERATIONS; ++i) {
		cout << "Original problem @ " << vec_time_o[i] << "s, lb: " << vec_lb_o[i] << ", ub_mean:" << vec_ub_o_m[i] << ", ub_bound:" << vec_ub_o_b[i] << endl;
	}
    
    for (unsigned int i = 0; i < ITERATIONS; ++i) {
        cout << "Perturbed problem: " << vec_weights_c[i] << endl;
    }

    for (unsigned int i = 0; i < ITERATIONS; ++i) {
        cout << "Perturbed problem @ " << vec_time_c[i] << "s, lb: " << vec_lb_c[i] << ", ub_mean:" << vec_ub_c_m[i] << ", ub_bound:" << vec_ub_c_b[i] << endl;
    }
    for (unsigned int i = 0; i < ITERATIONS; ++i) {
        cout << "Contamination bound @ " << vec_time_der[i] << "s, mean:" << vec_der_c_mean[i] << ", bound:" << vec_der_c_bound[i] << endl;
    }

	//clean up
	delete model;
	delete model_cont;

	//even when debugging do not end
	char c;
	cin >> c;

	return 0;
}


/* obsolete parts of code, needs to be revisited
//variance
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
	for (double lambda = 0.01; lambda < 1.00; lambda += 0.01) {
		mean_file << "\t" << lambda;
		variance_file << "\t" << lambda;
	}
	mean_file << endl;
	variance_file << endl;
	for (double prob = 0.01; prob < 1.00; prob += 0.01) {
		mean_file << prob;
		variance_file << prob;
		for (double lambda = 0.01; lambda < 1.00; lambda += 0.01) {
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
	if (append != "") {
		in_file_name += append;
		mean_file_name += append;
		variance_file_name += append;
	}
	in_file_name += ".txt";
	mean_file_name += ".txt";
	variance_file_name += ".txt";
	datafile.open(in_file_name);
	stringstream str;
	while (getline(datafile, line)) {
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
	for (double lambda = 0.1; lambda < 1.00; lambda += 0.10) {
		mean_file << "\t" << lambda;
		variance_file << "\t" << lambda;
	}
	mean_file << endl;
	variance_file << endl;
	for (double prob = 0.01; prob < 1.00; prob += 0.01) {
		mean_file << prob;
		variance_file << prob;
		//for(double lambda = 0.01; lambda < 1.00; lambda += 0.01) {
		for (double lambda = 0.1; lambda < 1.00; lambda += 0.10) {
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
*/
