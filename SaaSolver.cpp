
#include "SaaSolver.h"
#include "GamsWrapper.h"
#include "Helper.h"
#include <boost/algorithm/string/join.hpp>
#include <map>
using namespace boost;

const string SaaSolver::GAMS_VAR_OUT = "gams_var_saa.out";
const string SaaSolver::GAMS_SCRIPT = "risk_saa.gms";
const string SaaSolver::GAMS_EXE_PATH = "C:\\Program Files\\GAMS\\gams.exe";
const string SaaSolver::GAMS_RISK_VAR = "totalrisk";
const string SaaSolver::GAMS_PROFIT_VAR = "totalprofit";
const string SaaSolver::GAMS_MODEL_VAR = "modelstatus";
const string SaaSolver::GAMS_SOLVER_VAR = "solverstatus";
const string SaaSolver::GAMS_OBJECTIVE_VAR = "objective";

SaaSolver::SaaSolver(const ScenarioModel *model)
: Solver(model)
{
	tree_ = 0;

	if(model_->GetParam().risk_measure != RISK_CVAR_SUM) {
		throw SaaSolverException("Only CVaR risk measure is implemented.");
	}
}

SaaSolver::~SaaSolver(void)
{
}

void SaaSolver::Solve(mat &weights, double &objective) {
	tree_ = this->GetTree();

	//prepare the script
	ofstream writefile;
	writefile.open(GAMS_SCRIPT.c_str(), ios::out | ios::trunc);
	if(!writefile.is_open()) {
		throw SaaSolverException("Failed writing to script file");
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

string SaaSolver::SetsGams() {
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

	str << "Set s Scenarios /";
	str << "1*" << tree_->ScenarioCount();
	str << " / ;" << endl;
	str << "Alias(s, s1);" << endl;

	str << "Set st Scenario to Stage mapping /" << endl;
	for(SCENINDEX s = 0; s < tree_->ScenarioCount(); ++s) {
		str << setw(8) << (*tree_)(s).GetNumber() << "." << (*tree_)(s).GetStage() << endl;
	}
	str << " / ;" << endl;

	str << "Set sp Scenario to its parent mapping /" << endl;
	for(SCENINDEX s = 0; s < tree_->ScenarioCount(); ++s) {
		
		str << setw(8) << (*tree_)(s).GetNumber() << ".";
		if((*tree_)(s).HasParent()) { //exclude root
			str << (*tree_)(s).GetParent().GetNumber();
		}
		else {
			str << (*tree_)(s).GetNumber();
		}
		str << endl;
	}
	str << " / ;" << endl;

	return str.str();
}

string SaaSolver::ParametersGams() {
	//parameters - scenario values
	stringstream str; 
	
	str << "Table sc(s,a) generated scenarios" << endl;
	str << setprecision(10) << setfill(' ');
	str << setw(8) << " ";
	for(unsigned int a = 0; a < model_->GetAssetsCount(); ++a) {
		str << " " << setw(25) << model_->GetAssets()[a];
	}
	str << endl;
	for(SCENINDEX s = 0; s < tree_->ScenarioCount(); ++s) {
		str << setw(8) << (*tree_)(s).GetNumber();
		for(unsigned int i = 0; i < tree_->GetNodeSize(); ++i) {
			str << " " << setw(25) << (*tree_)(s).GetValues()[i];
		}
		str << endl;
	}
	str << " ;" << endl;

	//stage scenarios
	str << "Parameter ts(t) Scenario count for each stage /" << endl;
	for(unsigned int i = 0; i < model_->GetStages().size(); ++i) {
		str << setw(8) << model_->GetStages()[i] << " " << setw(8) << tree_->ScenarioCountStage(model_->GetStages()[i]) << endl;
	}
	str << " /;" << endl;

	//rsik parameters
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
	str << "Scalar stages / " << tree_->StageCount() << " /;" << endl;

	return str.str();
}

string SaaSolver::VariablesGams() {
	stringstream str; 
	str << "Variable w(s, a) weights of portfolio in each scenario;" << endl;
	str << "Positive variable w;" << endl;	
	str << "Variable p(t) profit at each stage" << endl;
	str << "Variable r(t) risk measure in stages;" << endl;
	str << "Variable rt total risk to minimize;" << endl;
	str << "Variable pt total profit to maximize;" << endl;
	str << "Variable rht(t) variable to help calculate CVaR;" << endl;
	str << "Variable rhs(s) variable to help calculate CVaR;" << endl;
	str << "Variable f objective function to minimize;" << endl;
	str << "Positive variable rhs;" << endl;

	return str.str();
}

string SaaSolver::EquationsGams() {
	stringstream str; 

	str << "Equation weights0(s, t) restricted to sum up to our initial capital;" << endl;
	str << "Equation weightst(s, t) restricted to sum up to our actual capital;" << endl;
	str << "Equation profit(t) to be expected at each stage;" << endl;
	str << "Equation profittotal profit to be achieved at the last stage" << endl;
	str << "Equation risk(t) risk associated to each stage;" << endl;
//	str << "Equation subject(t) total risk to be minimized;" << endl;
	str << "Equation risktotal total risk to be minimized;" << endl;
	str << "Equation objective specifies objective to be minimized;" << endl;

	str << "Equation riskhelps(s) helping equations to calculate CVaR;" << endl;

	str << "weights0(s, t)$(st(s, t)$(ord(t) = 1)) .. sum(a, w(s, a)) =e= 1;" << endl; //init capital is one
	str << "weightst(s, t)$(st(s, t)$(ord(t) > 1))  .. sum(a, w(s, a)) =e= sum(s1$sp(s, s1), sum(a1, w(s1, a1) * sc(s, a1)));" << endl;
	str << "profit(t) .. p(t) =e= (1/ts(t)) * sum(s$st(s,t), sum(a, w(s, a)));" << endl;
	str << "profittotal .. pt =e= sum(t, expectationcoefficient(t) * p(t));" << endl;
	str << "risk(t) .. r(t) =e= rht(t) + 1/( ts(t) * (1 - confidence)) * sum(s$st(s, t), rhs(s));" << endl;
//	str << "subject(t)$(ord(t) > 1) .. c =g= r(t);" << endl;
	str << "riskhelps(s) .. rhs(s) =g= - sum(a, w(s,a)) - sum(t$st(s,t), rht(t));" << endl;
	str << "risktotal .. rt =e= sum(t, riskcoefficient(t) * r(t));" << endl;
	str << "objective .. f =e= -pt + rt;" << endl;
	//TODO: add discount factor
	
	return str.str();
}

string SaaSolver::SolveGams() {
	stringstream str; 
	str << "Option decimals=8;" << endl;    
	str << "Option reslim=10000;" << endl;
	str << "Option iterlim=1000000;" << endl;
	str << "Model riskmodel /all/;" << endl;
	str << "Solve riskmodel using lp minimizing f;" << endl;
    str << "File results /" << GAMS_VAR_OUT << "/;" << endl;
	str << "Put results;" << endl;
	str << "Put '" << GAMS_OBJECTIVE_VAR <<"', @30 f.l:20:18 /;" << endl;
	str << "Put '" << GAMS_RISK_VAR <<"', @30 rt.l:20:18 /;" << endl;
	str << "Put '" << GAMS_PROFIT_VAR <<"', @30 pt.l:20:18 /;" << endl;
	str << "Put '" << GAMS_MODEL_VAR <<"', @30 riskmodel.modelstat:8:0 /;"  << endl;
	str << "Put '" << GAMS_SOLVER_VAR <<"', @30 riskmodel.solvestat:8:0 /;"  << endl;
	str << "Loop(a, Loop(s, Put$sp(s, s) a.tl, @30 w.l(s, a):20:18 /));" << endl;

	return str.str();
}

void SaaSolver::ParseOutputGams(mat &weights, double &objective) {
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
	if(lexical_cast<int>(values[GAMS_MODEL_VAR]) != 1) {
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

void SaaSolver::GetStageSamples(std::vector<unsigned int> &stage_samples) {
	if(model_->GetStagesCount() <= 0) {
		return;
	}
	stage_samples.push_back(1); //fixed root
	//TODO: reconsider
	for(unsigned int stage = 2; stage <= model_->GetStagesCount(); ++stage) {
#ifdef _DEBUG
		stage_samples.push_back(20);
#else
		stage_samples.push_back(100);
#endif
	}
}