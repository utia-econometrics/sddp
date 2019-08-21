#include "AssetAllocationModel.h"
#include "NormalDistribution.h"

#define DEBUG_DISTRIBUTION 0

double AssetAllocationModel::EPSILON = 0.000001;

AssetAllocationModel::AssetAllocationModel(unsigned int stages, const vector<string> &assets, RiskParameters param, const colvec &mu, const mat &sigma)
: ScenarioModel(stages)
{
	mu_ = mu;
	sigma_ = sigma;

	Init(stages, assets, param);
}

AssetAllocationModel::AssetAllocationModel(unsigned int stages, const vector<string> &assets, RiskParameters param, const mat &sample)
: ScenarioModel(stages)
{
	mat estimator = log(sample);
	NormalDistribution normal(estimator);
	normal.MLEstimate();
	
	mu_ = normal.GetMu();
	sigma_ = normal.GetSigma();

	Init(stages, assets, param);
}

void AssetAllocationModel::Init(unsigned int stages, const vector<string> &assets, RiskParameters &param) {
	assets_ = assets;
	param_ = param;
#ifdef _DEBUG
	if (param_.expectation_coefficients.size() != stages) {
		throw ScenarioModelException("Invalid size of expectation coefficients vector");
	}
	if (param_.risk_coefficients.size() != stages) {
		throw ScenarioModelException("Invalid size of risk coefficients vector");
	}

	//the coefficients for the first stage should be zero (deterministic!)
	if ((stages > 0) && (abs(param_.expectation_coefficients[0]) > EPSILON)) {
		throw ScenarioModelException("Expectation coefficient for the first stage has to be zero");
	}
	if ((stages > 0) && (abs(param_.risk_coefficients[0]) > EPSILON)) {
		throw ScenarioModelException("Risk coefficient for the first stage has to be zero");
	}

	if (param_.confidence > param_.confidence_other) {
		throw ScenarioModelException("Invalid configuration, second confidence level has to be higher than the first.");
	}
#endif
}

AssetAllocationModel::~AssetAllocationModel(void)
{
}

colvec AssetAllocationModel::GetMu() const {
	return mu_;
}

void AssetAllocationModel::SetMu(colvec mu) {
    mu_ = mu;
}

mat AssetAllocationModel::GetSigma() const {
	return sigma_;
}

void AssetAllocationModel::SetSigma(mat sigma) {
    sigma_ = sigma;
}

Distribution* AssetAllocationModel::GetDistribution(unsigned int stage, unsigned int state) const {
#if DEBUG_DISTRIBUTION == 1
	if(true) {
#else
	if(stage == 1) {
#endif
		//fixed root e^0 = 1
		colvec mu = zeros(GetAssetsCount());
		mat sigma = zeros(GetAssetsCount(), GetAssetsCount());
		return new NormalDistribution(mu, sigma);
	}
	if (GetStageDependence() == STAGE_INDEPENDENT) {
		return new NormalDistribution(mu_, sigma_);
	}
	else if (GetStageDependence() == MARKOV) {
		if (state == 1) { //no-crisis
			return new NormalDistribution(mu_, sigma_);
		}
		else if (state == 2) { //crisis
			return new NormalDistribution(mu_, sigma_ * param_.markov_crisis_variance_factor);
		}
		else {
			throw ScenarioModelException("Unknown state.");
		}
	}
	else {
		throw ScenarioModelException("Unknown dependence model.");
	}
}

unsigned int AssetAllocationModel::GetDecisionSize(unsigned int stage) const {
	//we need co compute weights for each asset is our model and two auxilliary variables - VaR level and VaR level2
	return GetAssetsCount() + 2;
}

void AssetAllocationModel::Evaluate(double *value) const {
	colvec cur_value(value, GetAssetsCount(), false);
	cur_value = exp(cur_value);
}

StageDependence AssetAllocationModel::GetStageDependence() const {
	return param_.stage_dependence;
}

unsigned int AssetAllocationModel::GetStatesCountStage(unsigned int stage) const {
	if (stage == 1) {
		//fixed root
		return 1;
	}
	if (GetStageDependence() == STAGE_INDEPENDENT) {
		return 1;
	}
	else if (GetStageDependence() == MARKOV) {
		return 2; // crisis x non-crisis
	}
	else {
		throw ScenarioModelException("Unknown dependence model.");
	}
}

mat AssetAllocationModel::GetTransitionProbabilities(unsigned int stage) const {
	if (GetStageDependence() == STAGE_INDEPENDENT) {
		return ones(1);
	}
	else if (GetStageDependence() == MARKOV) {
		if (stage == 1) {
			//we suppose root is in non-crisis, fixed state == 1
			mat prob(1, 2);
			prob(0, 0) = 1 - param_.markov_to_crisis_probability;
			prob(0, 1) = param_.markov_to_crisis_probability;
			return prob;
		}
		else {
			//make 2x2 matrix of transitions
			mat prob(2, 2);
			prob(0, 0) = 1 - param_.markov_to_crisis_probability;
			prob(0, 1) = param_.markov_to_crisis_probability;
			prob(1, 0) = param_.markov_from_crisis_probability;
			prob(1, 1) = 1 - param_.markov_from_crisis_probability;
			return prob;
		}
	}
	else {
		throw ScenarioModelException("Unknown dependence model.");
	}
}

double AssetAllocationModel::GetRecourseLowerBound(unsigned int stage) const {
	if (stage == GetStagesCount()) {
		return 0.0; //last stage has fixed zero recourse
	}
	else {
		return -2.0 * (GetStagesCount() - stage);
	}
}

double AssetAllocationModel::GetRecourseUpperBound(unsigned int stage) const {
	if (param_.risk_measure == RISK_CVAR_NESTED) {
		return 0.0;
	}
	else {
		double conf_inv = 1 - param_.confidence_other;
		//get the max lower cvar bound and multiply it by one over atom size
		return -GetRecourseLowerBound(stage) / conf_inv;
	}
}

void AssetAllocationModel::BuildCoinModel(CoinModelWrapper * coin_model, unsigned int stage, const double *prev_decisions, const double *scenario, vector<string> &decision_vars, vector<string> &dual_constraints) const {
	//assets count N
	unsigned int assets = GetAssetsCount();
	bool root_node = (stage == 1);
	bool last_stage = (stage == GetStagesCount());
	double conf_inv = 1 - param_.confidence;
	double conf_inv_other = 1 - param_.confidence_other;

	//var x1, ..., xN = weights 1 .. N
	for (unsigned int i = 1; i <= assets; ++i) {
		stringstream str_a;
		str_a << "x_" << stage << "_" << i;
		string var_name = str_a.str();
		//add all decision vars to the model and output to solver
		decision_vars.push_back(var_name);
		coin_model->AddVariable(var_name);
		//weights are positive
		coin_model->AddLowerBound(var_name, 0.0);
		//max profit = min negative loss
		if (!root_node) {
			switch (param_.risk_measure) {
			case RISK_CVAR_NESTED:
				coin_model->AddObjectiveCoefficient(var_name, -1.0);
				break;
			case RISK_CVAR_MULTIPERIOD:
				coin_model->AddObjectiveCoefficient(var_name, -1.0 * param_.expectation_coefficients[stage - 1]);
				break;
			}
		}
	}

	//var c = cvar part (positive value)
	if (!root_node && (param_.risk_measure == RISK_CVAR_MULTIPERIOD)) {
		coin_model->AddVariable("c");
		coin_model->AddLowerBound("c", 0); //positive part
										   //objective with risk aversion coef lambda
		coin_model->AddObjectiveCoefficient("c", param_.risk_coefficients[stage - 1] / conf_inv);
	}

	//var c_other = cvar part (positive value)
	if (!root_node && (param_.risk_measure == RISK_CVAR_MULTIPERIOD)) {
		coin_model->AddVariable("c_other");
		coin_model->AddLowerBound("c_other", 0); //positive part
												 //objective with risk aversion coef lambda
		coin_model->AddObjectiveCoefficient("c_other", param_.risk_coefficients_other[stage - 1] / conf_inv_other);
	}

	//var var = variable to calculate CVaR = VaR level, right after the allocations in the decision vector
	string dec_var_name = "var";
	//add all decision vars to the model and output to solver
	decision_vars.push_back(dec_var_name);
	coin_model->AddVariable(dec_var_name);
	coin_model->AddLowerBound(dec_var_name, GetRecourseLowerBound(stage));
	coin_model->AddUpperBound(dec_var_name, GetRecourseUpperBound(stage));
	//objevtive according to the risk aversion lambda
	if (!last_stage) {
		//last stage has no recourse
		coin_model->AddObjectiveCoefficient(dec_var_name, param_.risk_coefficients[stage]);
	}

	//var var_other = variable to calculate CVaR = VaR level, right after the first var level in the decision vector
	string dec_var_name_other = "var_other";
	//add all decision vars to the model and output to solver
	decision_vars.push_back(dec_var_name_other);
	coin_model->AddVariable(dec_var_name_other);
	coin_model->AddLowerBound(dec_var_name_other, GetRecourseLowerBound(stage));
	coin_model->AddUpperBound(dec_var_name_other, GetRecourseUpperBound(stage));
	//objevtive according to the risk aversion lambda
	if (!last_stage) {
		//last stage has no recourse
		coin_model->AddObjectiveCoefficient(dec_var_name_other, param_.risk_coefficients_other[stage]);
	}

	if (!root_node) {
		//var trans = transaction costs absolute value of stocks position difference
		for (unsigned int i = 1; i <= assets; ++i) {
			stringstream str;
			str << "trans_" << stage << "_" << i;
			string var_name = str.str();
			coin_model->AddVariable(var_name);
			coin_model->AddLowerBound(var_name, 0.0);
		}
	}

	//constraint capital = sum of the weights equals initial capital one
	coin_model->AddConstraint("capital");
	double transaction_coef = param_.transaction_costs;
	for (unsigned int i = 1; i <= assets; ++i) {
		string var_asset = decision_vars[i - 1];
		coin_model->AddConstraintVariable("capital", var_asset, 1);
		if (!root_node) {
			stringstream str;
			str << "trans_" << stage << "_" << i;
			string var_absolute = str.str();
			coin_model->AddConstraintVariable("capital", var_absolute, transaction_coef);
		}
	}
	double capital;
	if (root_node) {
		//no parent, init the capital with 1
		capital = 1.0;
	}
	else {
		//sum up the capital under this scenario
		capital = CalculateCapital(prev_decisions, scenario);
	}
	coin_model->AddConstrainBound("capital", EQUAL_TO, capital);
	dual_constraints.push_back("capital"); //we need dual of this

	//constraint trans = absolute values for transaction costs
	if (!root_node) { //stage >= 2
		for (unsigned int i = 1; i <= assets; ++i) {
			string var_asset = decision_vars[i - 1];
			stringstream str;
			str << "trans_" << stage << "_" << i;
			string var_absolute = str.str();
			str.str("");
			str << "trans_pos_" << stage << "_" << i;
			string cons_name_pos = str.str();
			str.str("");
			str << "trans_neg_" << stage << "_" << i;
			string cons_name_neg = str.str();

			//current position in asset i
			double parent_value = scenario[i - 1] * prev_decisions[i - 1]; //decisions and scenarios have the same order here, decisions start with allocations

			//x_t - z <= x_t-1 .. from parent solution
			coin_model->AddConstraint(cons_name_pos);
			coin_model->AddConstraintVariable(cons_name_pos, var_asset, 1);
			coin_model->AddConstraintVariable(cons_name_pos, var_absolute, -1);
			coin_model->AddConstrainBound(cons_name_pos, LOWER_THAN, parent_value);

			//-x_t - z <= -x_t-1 .. from parent solution
			coin_model->AddConstraint(cons_name_neg);
			coin_model->AddConstraintVariable(cons_name_neg, var_asset, -1);
			coin_model->AddConstraintVariable(cons_name_neg, var_absolute, -1);
			coin_model->AddConstrainBound(cons_name_neg, LOWER_THAN, -parent_value);

			dual_constraints.push_back(cons_name_pos); //we need dual of this
			dual_constraints.push_back(cons_name_pos); //we need dual of this
		}
	}

	//constraint the positive part "c" of the CVaR variable
	if (!root_node && (param_.risk_measure == RISK_CVAR_MULTIPERIOD)) {
		coin_model->AddConstraint("cvar");
		coin_model->AddConstraintVariable("cvar", "c", 1.0); //the varible itself
		for (unsigned int i = 0; i < assets; ++i) {
			string var_name = decision_vars[i];
			coin_model->AddConstraintVariable("cvar", var_name, 1.0); //-c transp x
		}
		double parent_var = prev_decisions[assets]; //var is right after the allocations in decision vector
		coin_model->AddConstrainBound("cvar", GREATER_THAN, -parent_var); //previous value at risk
		dual_constraints.push_back("cvar"); //we need dual of this
	}


	//constraint the positive part "c_other" of the CVaR variable
	if (!root_node && (param_.risk_measure == RISK_CVAR_MULTIPERIOD)) {
		coin_model->AddConstraint("cvar_other");
		coin_model->AddConstraintVariable("cvar_other", "c_other", 1.0); //the varible itself
		for (unsigned int i = 0; i < assets; ++i) {
			string var_name = decision_vars[i];
			coin_model->AddConstraintVariable("cvar_other", var_name, 1.0); //-c transp x
		}
		double parent_var_other = prev_decisions[assets + 1]; //var_other is right after var in the decisions vector
		coin_model->AddConstrainBound("cvar_other", GREATER_THAN, -parent_var_other); //previous value at risk
		dual_constraints.push_back("cvar_other"); //we need dual of this
	}
}

void AssetAllocationModel::FillSubradient(unsigned int stage, const double *prev_decisions, const double *scenario, double objective, const double *duals, double &recourse, double *subgradient) const {
	unsigned int assets = GetAssetsCount();
	bool root_node = (stage == 1);
	bool last_stage = (stage == GetStagesCount());

	//needed for recourse, from previous stage
	double expec_c = param_.expectation_coefficients[stage - 1];
	double risk_c = param_.risk_coefficients[stage - 1];
	double risk_c_other = param_.risk_coefficients_other[stage - 1];
	double conf_inv = 1 - param_.confidence;
	double conf_inv_other = 1 - param_.confidence_other;

	if (root_node) {
		switch (param_.risk_measure) {
		case RISK_CVAR_NESTED:
			//expectation and CVaR part
			recourse = expec_c * objective;
			break;
		case RISK_CVAR_MULTIPERIOD:
			recourse = objective;
			break;
		}

		return; //no real calculation
	}

	//Shapiro (2001) 4.27
	double parent_var = prev_decisions[assets]; //var is right after the allocations in decision vector
	double parent_var_other = prev_decisions[assets + 1]; //var_other is right after var in the decisions vector
	switch (param_.risk_measure) {
		case RISK_CVAR_NESTED:
			//expectation and CVaR part
			recourse = expec_c * objective;
			if (objective > parent_var) {
				//positive part of CVaR calculation
				recourse += (risk_c / conf_inv) * (objective - parent_var);
			}
			if (objective > parent_var_other) {
				//positive part of CVaR calculation
				recourse += (risk_c_other / conf_inv_other) * (objective - parent_var_other);
			}
			break;
		case RISK_CVAR_MULTIPERIOD:
			recourse = objective;
			break;
	}

	//Shapiro SPBook Proposition 2.2 .. T = (-price1, ... -price N) ; W = (1, ..., 1), h = 0
	unsigned int dual_index = 0; //at first place;
	double dual_cap = duals[dual_index++];
	for (unsigned int i = 1; i <= assets; ++i) {
		double subgrad = 0.0;
		double value = scenario[i - 1];

		//the capital part
		subgrad += dual_cap * value;

		//iterate through duals
		double dual_trans_pos = duals[dual_index++];
		double dual_trans_neg = duals[dual_index++];

		subgrad += dual_trans_pos * value;
		subgrad -= dual_trans_neg * value;

		switch (param_.risk_measure) {
		case RISK_CVAR_NESTED:
			subgradient[i - 1] = expec_c * subgrad;
			if (objective > parent_var) {
				subgradient[i - 1] += (risk_c / conf_inv) * subgrad;
			}
			if (objective > parent_var_other) {
				subgradient[i - 1] += (risk_c_other / conf_inv_other) * subgrad;
			}
			break;
		case RISK_CVAR_MULTIPERIOD:
			subgradient[i - 1] = subgrad;
			break;
		}
	}

	//subgradient for the var level
	//Shapiro (2011) 4.29
	switch (param_.risk_measure) {
		case RISK_CVAR_NESTED:
			//var and var other subgradients are put right after asset allocations
			subgradient[assets] = 0.0;
			subgradient[assets + 1] = 0.0;
			if (objective > parent_var) {
				subgradient[assets] -= (risk_c / conf_inv);
			}
			if (objective > parent_var_other) {
				subgradient[assets + 1] -= (risk_c_other / conf_inv_other);
			}
			break;
		case RISK_CVAR_MULTIPERIOD:
			//var and var other subgradients are put at the last position and only for multiperiod
			subgradient[assets] = -duals[dual_index++];
			subgradient[assets + 1] = -duals[dual_index++];
			break;
	}
}

#ifndef _DEBUG
void AssetAllocationModel::BuildCplexModel(IloEnv &env, IloModel &model, IloExpr &objective, unsigned int stage, const double *prev_decisions, const double *scenario, vector<IloNumVar> &decision_vars, vector<IloRange> &dual_constraints) const {
	//assets count N
	unsigned int assets = GetAssetsCount();
	bool root_node = (stage == 1);
	bool last_stage = (stage == GetStagesCount());
	double conf_inv = 1 - param_.confidence;
	double conf_inv_other = 1 - param_.confidence_other;

	IloNumVarArray x(env, assets);
	for (unsigned int i = 0; i < assets; ++i) {
		//zero lower bound
		x[i] = IloNumVar(env, 0.0);

		//max profit = min negative loss
		if (!root_node) {
			switch (param_.risk_measure) {
			case RISK_CVAR_NESTED:
				objective += -1 * x[i];
				break;
			case RISK_CVAR_MULTIPERIOD:
				objective += -1 * param_.expectation_coefficients[stage - 1] * x[i];
				break;
			}
		}

		decision_vars.push_back(x[i]);
	}
	model.add(x);

	//var c = positive value in the cvar
	IloNumVar c(env, 0.0);
	if (!root_node && (param_.risk_measure == RISK_CVAR_MULTIPERIOD)) {
		//objective with risk aversion coef lambda
		objective += param_.risk_coefficients[stage - 1] / conf_inv * c;
		model.add(c);
	}

	//var c_other = positive value in the cvar
	IloNumVar c_other(env, 0.0);
	if (!root_node && (param_.risk_measure == RISK_CVAR_MULTIPERIOD)) {
		//objective with risk aversion coef lambda
		objective += param_.risk_coefficients_other[stage - 1] / conf_inv_other * c_other;
		model.add(c_other);
	}

	//var var = variable to calculate CVaR = VaR level
	IloNumVar var(env, GetRecourseLowerBound(stage), GetRecourseUpperBound(stage));
	//objective according to the risk aversion lambda
	if (!last_stage) {
		//last stage has no recourse
		objective += param_.risk_coefficients[stage] * var;
	}
	model.add(var);
	decision_vars.push_back(var);

	//var var_other = variable to calculate CVaR = VaR level
	IloNumVar var_other(env, GetRecourseLowerBound(stage), GetRecourseUpperBound(stage));
	//objective according to the risk aversion lambda
	if (!last_stage) {
		//last stage has no recourse
		objective += param_.risk_coefficients_other[stage] * var_other;
	}
	model.add(var_other);
	decision_vars.push_back(var_other);

	//transaction costs dummy variable
	IloNumVarArray trans(env, assets);
	for (unsigned int i = 0; i < assets; ++i) {
		//zero lower bound
		trans[i] = IloNumVar(env, 0.0);
	}
	if (!root_node) {
		model.add(trans);
	}

	//constraint capital = sum of the weights equals initial capital one
	IloExpr cap_expr(env);
	double transaction_coef = param_.transaction_costs;
	for (unsigned int i = 0; i < assets; ++i) {
		cap_expr += x[i];
		if (!root_node) {
			cap_expr += transaction_coef * trans[i];
		}
	}
	double total_capital;
	if (root_node) {
		//no parent, init the capital with 1
		total_capital = 1.0;
	}
	else {
		//sum up the capital under this scenario
		total_capital = CalculateCapital(prev_decisions, scenario);
	}
	IloRange capital(env, total_capital, cap_expr, total_capital);
	model.add(capital);
	dual_constraints.push_back(capital);

	//constraint transaction costs
	IloRangeArray costs_pos(env);
	IloRangeArray costs_neg(env);
	if (!root_node) {
		for (unsigned int i = 0; i < assets; ++i) {
			//current position in asset i
			double parent_value = scenario[i] * prev_decisions[i]; //decisions and scenarios have the same order here, decisions start with allocations

			//positive and negative part of linearization for absolute value
			IloExpr cost_expr_pos(env);
			IloExpr cost_expr_neg(env);
			cost_expr_pos += x[i];
			cost_expr_pos += -1 * trans[i];
			cost_expr_neg += -1 * x[i];
			cost_expr_neg += -1 * trans[i];
			costs_pos.add(IloRange(env, cost_expr_pos, parent_value));
			costs_neg.add(IloRange(env, cost_expr_neg, -parent_value));

			dual_constraints.push_back(costs_pos[i]);
			dual_constraints.push_back(costs_neg[i]);
		}
		model.add(costs_pos);
		model.add(costs_neg);
	}

	//contraint the positive part "c" of the CVaR value
	IloExpr cvar_expr(env);
	IloRange cvar;
	if (!root_node && (param_.risk_measure == RISK_CVAR_MULTIPERIOD)) {
		cvar_expr += c; //the varible itself
		for (unsigned int i = 0; i < assets; ++i) {
			cvar_expr += x[i]; //-c transp x
		}
		double parent_var = prev_decisions[assets]; //var is right after the allocations in decision vector
		cvar = IloRange(env, -parent_var, cvar_expr); //previous value at risk
		model.add(cvar);
		dual_constraints.push_back(cvar);
	}

	//contraint the positive part "c_other" of the CVaR value
	IloExpr cvar_expr_other(env);
	IloRange cvar_other;
	if (!root_node && (param_.risk_measure == RISK_CVAR_MULTIPERIOD)) {
		cvar_expr_other += c_other; //the varible itself
		for (unsigned int i = 0; i < assets; ++i) {
			cvar_expr_other += x[i]; //-c transp x
		}
		double parent_var_other = prev_decisions[assets + 1]; //var_other is right after var in the decisions vector
		cvar_other = IloRange(env, -parent_var_other, cvar_expr_other); //previous value at risk
		model.add(cvar_other);
		dual_constraints.push_back(cvar_other);
	}

}
#endif

double AssetAllocationModel::CalculateUpperBound(unsigned int stage, const double *prev_decisions, const double *decisions, const double *scenario, double recourse_estimate, bool cut_tail) const {
	//constants
	double conf_inv = 1 - param_.confidence;
	double conf_inv_other = 1 - param_.confidence_other;
	unsigned int assets = GetAssetsCount();
	bool root_node = (stage == 1);
	bool last_stage = (stage == GetStagesCount());
	

	//node value in money
	double node_value = -CalculateNodeValue(decisions);

	//coefficients from current stage
	double exp_c = param_.expectation_coefficients[stage - 1];
	double risk_c = param_.risk_coefficients[stage - 1];
	double risk_c_other = param_.risk_coefficients_other[stage - 1];

	double var = decisions[assets]; //var is right after the allocations in decision vector
	double var_other = decisions[assets + 1]; //var_other is right after var in the decisions vector

	//coefficients from previous stage
	double next_exp_c = 0;
	double next_risk_c = 0;
	double next_risk_c_other = 0;

	if (!last_stage) {
		next_exp_c = param_.expectation_coefficients[stage];
		next_risk_c = param_.risk_coefficients[stage];
		next_risk_c_other = param_.risk_coefficients_other[stage];
	}

	if (root_node) {
		double root_value = recourse_estimate;
		root_value += -exp_c * node_value; //intenationaly make it zero even for nested CVaR
		root_value += next_risk_c * var;
		root_value += next_risk_c_other * var_other;
		return root_value;
	}

	//VaR levels from "parent" - available, because we are not in root
	double prev_var = prev_decisions[assets]; //var is right after the allocations in decision vector
	double prev_var_other = prev_decisions[assets + 1]; //var_other is right after var in the decisions vector

	//taken from the future stages
	double total_value = 0.0;
	double recourse_value = 0.0;
	if (!last_stage) {
		total_value = recourse_estimate;
	}

	switch (param_.risk_measure) {
		case RISK_CVAR_NESTED:
			//taken from the future stages
			recourse_value = total_value;

			//calculate cost of current solutions
			recourse_value += node_value;

			if (!last_stage) {
				///account for the "u" variables for the next stage
				recourse_value += next_risk_c * var;
				recourse_value += next_risk_c_other * var_other;
			}

			//recourse function expectation constant
			total_value = exp_c * recourse_value;

			//calculate CVaR using the "u" variables from the previous stage
			if ((recourse_value > prev_var) && (!cut_tail)) {
				//if we can get rid of the tails then do
				total_value += (risk_c / conf_inv) * (recourse_value - prev_var);
			}
			if ((recourse_value > prev_var_other) && (!cut_tail)) {
				//if we can get rid of the tails then do
				total_value += (risk_c_other / conf_inv_other) * (recourse_value - prev_var_other);
			}
			
			return total_value;
		case RISK_CVAR_MULTIPERIOD:
			//calculate cost of current solutions
			total_value += exp_c * node_value;

			if (!last_stage) {
				//account for the "u" variables for the next stage
				total_value += next_risk_c * var;
				total_value += next_risk_c_other * var_other;
			}

			//calculate CVaR using the "u" variables from the previous stage
			if ((node_value > prev_var) && (!cut_tail)) {
				//if we can get rid of the tails then do
				total_value += (risk_c / conf_inv) * (node_value - prev_var);
			}
			if ((node_value > prev_var_other) && (!cut_tail)) {
				//if we can get rid of the tails then do
				total_value += (risk_c_other / conf_inv_other) * (node_value - prev_var_other);
			}

			return total_value;
	}

	return 0; //shall not happen
}

double AssetAllocationModel::CalculateNodeValue(const double *decisions) const {
	double sum = 0.0;
	for (unsigned int i = 0; i < GetAssetsCount(); ++i) {
		sum += decisions[i];
	}
	return sum;
}

double AssetAllocationModel::GetDiscountFactor(unsigned int stage) const {
	return param_.discount_factor;
}

double AssetAllocationModel::ApproximateDecisionValue(unsigned int stage, const double *prev_decisions, const double *scenario) const {
	return CalculateCapital(prev_decisions, scenario);
}

double AssetAllocationModel::GetTailAlpha(unsigned int stage) const {
	return param_.confidence;
}

double AssetAllocationModel::CalculateTailCutoff(unsigned int stage, double var_h) const {
	double cut_margin = var_h;
	cut_margin *= 1 + param_.transaction_costs; //sell all
	cut_margin /= 1 - param_.transaction_costs; //buy something else
	return cut_margin;
}

double AssetAllocationModel::CalculateCapital(const double *decisions, const double *scenario) const {
	double sum = 0.0;
	//first decisions are asset allocations
	for (unsigned int i = 0; i < GetAssetsCount(); ++i) {
		sum += decisions[i] * scenario[i];
	}
	return sum;
}
