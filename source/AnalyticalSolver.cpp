#include "AnalyticalSolver.h"
#include "GamsWrapper.h"
#include "Helper.h"
#include <boost/algorithm/string/join.hpp>
#include <map>
#include "alglib/ap.h"
#include "NormalDistribution.h"
using namespace boost;
using namespace alglib;

const string AnalyticalSolver::GAMS_VAR_OUT = "gams_var_analytical.out";
const string AnalyticalSolver::GAMS_SCRIPT = "risk_analytical.gms";
const string AnalyticalSolver::GAMS_EXE_PATH = "C:\\Program Files\\GAMS\\gams.exe";
const string AnalyticalSolver::GAMS_RISK_VAR = "totalrisk";
const string AnalyticalSolver::GAMS_PROFIT_VAR = "totalprofit";
const string AnalyticalSolver::GAMS_MODEL_VAR = "modelstatus";
const string AnalyticalSolver::GAMS_SOLVER_VAR = "solverstatus";
const string AnalyticalSolver::GAMS_OBJECTIVE_VAR = "objective";

AnalyticalSolver::AnalyticalSolver(const ScenarioModel *model)
: Solver(model)
{
	for(unsigned int t = 1; t <= model_->GetStagesCount(); ++t) {
		if(model_->GetTrueDistribution(t)->GetType() != DISTRIBUTION_NORMAL) {
			throw AnalyticalSolverException("Only normal distribution is implemented.");
		}
	}
	if(model_->GetParam().risk_measure != RISK_CVAR_SUM) {
		throw AnalyticalSolverException("Only CVaR SUM risk measure is implemented.");
	}
}

AnalyticalSolver::~AnalyticalSolver(void)
{
}

void AnalyticalSolver::Solve(mat &weights, double &objective) {
	//prepare the script
	ofstream writefile;
	writefile.open(GAMS_SCRIPT.c_str(), ios::out | ios::trunc);
	if(!writefile.is_open()) {
		throw AnalyticalSolverException("Failed writing to script file");
	}
	
	writefile << SetsGams();
	writefile << ParametersGams();
	writefile << VariablesGams();
	writefile << EquationsGams();
	writefile << SolveGams();
	writefile.close();

	//clean up
	remove(GAMS_VAR_OUT.c_str());

	//run GAMS
	GamsWrapper gw(GAMS_EXE_PATH);
	gw.RunGams(GAMS_SCRIPT.c_str());

	//process results
	ParseOutputGams(weights, objective);
}

string AnalyticalSolver::SetsGams() {
	//add the asstets set
	stringstream str;
	str << "$title risk_measures" << endl;
	
	str << "Set a Assets / ";
	str << algorithm::join(model_->GetAssets(), ", ");
	str << " / ;" << endl;
	str << "Alias(a, a1);" << endl;

	str << "Set t Stages / ";
	str << algorithm::join(Helper::ArrayToString(model_->GetStages()), ", ");
	str << " / ;" << endl;
	str << "Alias(t, t1);" << endl;

	return str.str();
}

string AnalyticalSolver::ParametersGams() {
	//parameters - scenario values
	stringstream str; 

	//mean in each stage
	str << "Table mu(t,a) expected outcomes at each stage for each asset" << endl;
	str << setprecision(10) << setfill(' ');
	str << setw(8) << " ";
	for(unsigned int a = 0; a < model_->GetAssetsCount(); ++a) {
		str << " " << setw(25) << model_->GetAssets()[a];
	}
	str << endl;
	for(unsigned int t = 1; t <= model_->GetStagesCount(); ++t) {
		str << setw(8) << t;
		for(unsigned int a = 0; a < model_->GetAssetsCount(); ++a) {
			str << " " << setw(25) << model_->GetTrueDistribution(t)->GetMu()(a);
		}
		str << endl;
	}
	str << " ;" << endl;

	//correlation in each stage
	str << "Table sigma(a, t, a1, t1) correlation matrix for each stage" << endl;
	for(unsigned int t = 1; t <= model_->GetStagesCount(); ++t) {
		str << setprecision(10) << setfill(' ');
		if(t == 1) {
			str << setw(8) << " ";
		}
		else {
			str << setw(8) << "+";
		}
		for(unsigned int a = 0; a < model_->GetAssetsCount(); ++a) {
			str << " " << setw(25) << model_->GetAssets()[a] << "." << t;
		}
		str << endl;
		for(unsigned int a = 0; a < model_->GetAssetsCount(); ++a) {
			str << setw(8) << model_->GetAssets()[a] << "." << t;
			for(unsigned int a1 = 0; a1 < model_->GetAssetsCount(); ++a1) {
				str << " " << setw(25) << model_->GetTrueDistribution(t)->GetSigma()(a, a1);
			}
			str << endl;
		}
	}
	str << " ;" << endl;
	
	//risk parameters
	str << "Parameter riskcoefficient(t) Coefficient for CVaR at each stage /" << endl;
	for(unsigned int i = 0; i < model_->GetStages().size(); ++i) {
		str << setw(8) << model_->GetStages()[i] << " " << setw(8) << model_->GetParam().risk_coefficients[i] << endl;
	}
	str << " /;" << endl;
	str << "Parameter expectationcoefficient(t) Coefficient for E[] at each stage /" << endl;
	for(unsigned int i = 0; i < model_->GetStages().size(); ++i) {
		str << setw(8) << model_->GetStages()[i] << " " << setw(8) << model_->GetParam().expectation_coefficients[i] << endl;
	}
	str << " /;" << endl;
	str << "Scalar discountfactor / " << model_->GetParam().discount_factor << " /;" << endl;
	str << "Scalar confidence / " << model_->GetParam().confidence << " /;" << endl;
	str << "Scalar stages / " << model_->GetStagesCount() << " /;" << endl;

	return str.str();
}

string AnalyticalSolver::VariablesGams() {
	stringstream str; 
	str << "Variable w(t, a) weights of portfolio in each stage;" << endl;
	str << "Positive variable w;" << endl;	
	str << "Variable p(t) profit at each stage" << endl;
	str << "Variable r(t) risk measure in stages;" << endl;
	str << "Variable rt total risk to minimize;" << endl;
	str << "Variable pt total profit to maximize;" << endl;
	str << "Variable f objective function to minimize;" << endl;

	return str.str();
}

string AnalyticalSolver::EquationsGams() {
	stringstream str; 

	str << "Equation weights0(t) restricted to sum up to our initial capital;" << endl;
	str << "Equation weightst(t) restricted to sum up to our actual capital;" << endl;
	str << "Equation profit(t) to be expected at each stage;" << endl;
	str << "Equation profittotal profit to be achieved at the last stage" << endl;
	str << "Equation risk(t) risk associated to each stage;" << endl;
	str << "Equation risktotal total risk to be minimized;" << endl;
	str << "Equation objective specifies objective to be minimized;" << endl;

	str << "weights0(t)$(ord(t) = 1) .. sum(a, w(t, a)) =e= 1;" << endl; //init capital is one
	str << "weightst(t)$(ord(t) > 1) .. sum(a, w(t, a)) =e= sum(t1$(ord(t1) = ord(t) - 1), sum(a, w(t1, a) * mu(t, a)));" << endl;
	str << "profit(t) .. p(t) =e= sum(a, w(t, a));" << endl;
	str << "profittotal .. pt =e= sum(t, expectationcoefficient(t) * p(t));" << endl;
	//TODO: FIX for other distributions
	double cvar_coef = exp(-1 * pow(model_->GetTrueDistribution(1)->InverseDistributionFunction(model_->GetParam().confidence), 2)/2)/((1 - model_->GetParam().confidence)*sqrt(2*pi()));
	str << "risk(t) .. r(t) =e= -p(t) + " << cvar_coef << " * sqrt(sum(t1$(ord(t1) = ord(t) - 1), sum((a, a1), w(t1, a) * sigma(a, t, a1, t) * w(t1, a1))));" << endl;
	str << "risktotal .. rt =e= sum(t, riskcoefficient(t) * r(t));" << endl;
	str << "objective .. f =e= -pt + rt;" << endl;
	//TODO: add discount factor
	
	return str.str();
}

string AnalyticalSolver::SolveGams() {
	stringstream str; 
	str << "Option decimals=8;" << endl;    
	str << "Option reslim=10000;" << endl;
	str << "Option iterlim=1000000;" << endl;
	str << "Model riskmodel /all/;" << endl;
	str << "Solve riskmodel using nlp minimizing f;" << endl;
    str << "File results /" << GAMS_VAR_OUT << "/;" << endl;
	str << "Put results;" << endl;
	str << "Put '" << GAMS_OBJECTIVE_VAR <<"', @30 f.l:20:18 /;" << endl;
	str << "Put '" << GAMS_RISK_VAR <<"', @30 rt.l:20:18 /;" << endl;
	str << "Put '" << GAMS_PROFIT_VAR <<"', @30 pt.l:20:18 /;" << endl;
	str << "Put '" << GAMS_MODEL_VAR <<"', @30 riskmodel.modelstat:8:0 /;"  << endl;
	str << "Put '" << GAMS_SOLVER_VAR <<"', @30 riskmodel.solvestat:8:0 /;"  << endl;
	str << "Loop(a, Put a.tl, @30 w.l('1', a):20:18 /);" << endl;

	return str.str();
}

void AnalyticalSolver::ParseOutputGams(mat &weights, double &objective) {
	//load raw data
	ifstream infile;
	infile.open(GAMS_VAR_OUT.c_str(), ios::in);
	if(!infile.is_open()) {
		throw SolverException("Failed opening GAMS output file");
	}
	string line;
	string key;
	string value;
	map<string, string> values;
	while(getline(infile,line)) {
		stringstream str;
		str << line;
		str >> key;
		str >> value;
		values[key] = value;
	}

	//check the solution status
	if((lexical_cast<int>(values[GAMS_MODEL_VAR]) != 1) && (lexical_cast<int>(values[GAMS_MODEL_VAR]) != 2)){ //locally optimal
		throw SolverException("Solution not optimal");
	}
	if(lexical_cast<int>(values[GAMS_SOLVER_VAR]) != 1) {
		throw SolverException("Solver status not normal");
	}

	//process required output
	objective = lexical_cast<double>(values[GAMS_OBJECTIVE_VAR]);
	weights.set_size(1, model_->GetAssetsCount());
	for(unsigned int i = 0; i < model_->GetAssetsCount(); ++i) {
		weights(i) = lexical_cast<double>(values[model_->GetAssets()[i]]);
	}
}

void AnalyticalSolver::GetStageSamples(std::vector<unsigned int> &stage_samples) {
	throw AnalyticalSolverException("This solver does not require sampling");
}
