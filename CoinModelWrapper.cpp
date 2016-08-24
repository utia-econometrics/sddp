#include "CoinModelWrapper.h"

CoinModelWrapper::CoinModelWrapper(void)
{
	model_built_ = false;
	model_solved_ = false;
	solver_interface_ = new OsiCbcSolverInterface();
}

CoinModelWrapper::~CoinModelWrapper(void)
{
	map<string, map<string, double> *>::iterator it;
	for(it = coinstraint_variables_.begin(); it != coinstraint_variables_.end(); ++it) {
		delete it->second;
	}
	delete solver_interface_;
}

void CoinModelWrapper::AddVariable(string variable) {
	variables_[variable] = true;
}

void CoinModelWrapper::AddLowerBound(string variable, double bound) {
	AssertValidVariable(variable);
	variable_lower_bounds_[variable] = bound;
}

void CoinModelWrapper::AddUpperBound(string variable, double bound) {
	AssertValidVariable(variable);
	variable_upper_bounds_[variable] = bound;
}

void CoinModelWrapper::AddObjectiveCoefficient(string variable, double value) {
	AssertValidVariable(variable);
	variable_objective_coefs_[variable] = value;
}
	
void CoinModelWrapper::AddConstraint(string constraint) {
	constraints_[constraint] = true;
	coinstraint_variables_[constraint] = new map<string, double>();
}
	
void CoinModelWrapper::AddConstraintVariable(string constraint, string variable, double value) {
	AssertValidConstraint(constraint);
	AssertValidVariable(variable);
	(*coinstraint_variables_[constraint])[variable] = value;
}

void CoinModelWrapper::AddConstrainBound(string constraint, BoundType bound_type, double value) {
	AssertValidConstraint(constraint);
	switch(bound_type) {
		case LOWER_THAN:
			constraint_upper_bounds_[constraint] = value;
			break;
		case EQUAL_TO:
			constraint_lower_bounds_[constraint] = value;
			constraint_upper_bounds_[constraint] = value;
			break;
		case GREATER_THAN:
			constraint_lower_bounds_[constraint] = value;
			break;
		default:
			throw CoinModelWrapperException("Specified bound type does not exist.");
	}
}

void CoinModelWrapper::AssertValidVariable(string variable) const {
#ifdef _DEBUG
	if(variables_.find(variable) == variables_.end()) {
		throw CoinModelWrapperException("Specified variable does not exist.");
	}
#endif
}

void CoinModelWrapper::AssertValidConstraint(string constraint) const {
#ifdef _DEBUG
	if(constraints_.find(constraint) == constraints_.end()) {
		throw CoinModelWrapperException("Specified constraint does not exist.");
	}
#endif
}

void CoinModelWrapper::AssertModelSolved() const {
#ifdef _DEBUG
	if(!model_solved_) {
		throw CoinModelWrapperException("Model was not solved yet.");
	}
#endif
}

void CoinModelWrapper::Fill_array(double *arr, map<string, bool> &indices, map<string, double> &values, double def_value) const {
	map<string, bool>::iterator itindices;
	map<string, double>::iterator itvalues;
	itvalues = values.begin();
	int index = 0;
	for(itindices = indices.begin(); itindices != indices.end(); ++itindices) {
		double value;
		if((itvalues != values.end()) && (itindices->first == itvalues->first)) {
			value = itvalues->second;
			++itvalues;
		}
		else {
			value = def_value;
		}
		arr[index++] = value;
	}
}

void CoinModelWrapper::BuildModel() {
	int nvars = variables_.size();
	int nconstr = constraints_.size();

	//init solver to get the infinity value, if we build another model clean up
	//HOTFIX while ->reset() does not work
	delete solver_interface_;
	solver_interface_ = new OsiCbcSolverInterface();
	

	//init vars for COIN-OR
	double *objective = new double[nvars];
	double *var_lb = new double[nvars];
	double *var_ub = new double[nvars];
	double *constr_lb = new double[nconstr];
	double *constr_ub = new double[nconstr];
	CoinPackedMatrix *matrix = new CoinPackedMatrix(false, 0, 0);
	matrix->setDimensions(0, nvars);

	//fill the objective function and variable bounds
	Fill_array(objective, variables_, variable_objective_coefs_, 0.0);
	Fill_array(var_lb, variables_, variable_lower_bounds_, -solver_interface_->getInfinity());
	Fill_array(var_ub, variables_, variable_upper_bounds_, solver_interface_->getInfinity());

	//fill the constraint matrix
	int *row_indices = new int[nvars];
	double *row_values = new double[nvars];
	map<string, map<string, double> *>::iterator itconstr;
	for(int i = 0; i < nvars; ++i) {
		row_indices[i] = i;
	}
	for(itconstr = coinstraint_variables_.begin(); itconstr != coinstraint_variables_.end(); ++itconstr) {
		Fill_array(row_values, variables_, *itconstr->second, 0.0);
		matrix->appendRow(nvars, row_indices, row_values);
	}
	delete row_values;
	delete row_indices;

	//fill constraint bounds
	Fill_array(constr_lb, constraints_, constraint_lower_bounds_, -solver_interface_->getInfinity());
	Fill_array(constr_ub, constraints_, constraint_upper_bounds_, solver_interface_->getInfinity());

	//build the model
	solver_interface_->loadProblem(*matrix, var_lb, var_ub, objective, constr_lb, constr_ub);
	solver_interface_->setHintParam(OsiDoReducePrint, true);

	//clean up
	delete constr_lb;
	delete constr_ub;
	delete matrix;
	delete objective;
	delete var_lb;
	delete var_ub;

	model_built_ = true;
}

void CoinModelWrapper::Solve() {
	if(!model_built_) {
		BuildModel();
	}
	solver_interface_->initialSolve();
	if(!solver_interface_->isProvenOptimal()) {
		cout << *this;
		throw CoinModelWrapperException("Optimal solution not found");
	}
	model_solved_ = true;
	ProcessSolution();
}

double CoinModelWrapper::GetSolution(string variable) {
	AssertValidVariable(variable);
	AssertModelSolved();
	return variable_solutions_[variable];
}

double CoinModelWrapper::GetDualPrice(string constraint) {
	AssertValidConstraint(constraint);
	AssertModelSolved();
	return dual_prices_[constraint];
}

double CoinModelWrapper::GetObjective() {
	AssertModelSolved();
	return objective_value_;
}

void CoinModelWrapper::ProcessSolution() {
	//objective
	objective_value_ = solver_interface_->getObjValue();
	
	//primal solution
	variable_solutions_.clear();
	const double * solution = solver_interface_->getColSolution();
	map<string, bool>::iterator it;
	int counter = 0;
	for(it = variables_.begin(); it != variables_.end(); ++it) {
		variable_solutions_[it->first] = solution[counter++];
	}

	//duals
	dual_prices_.clear();
	const double * duals = solver_interface_->getRowPrice();
	map<string, bool>::iterator itc;
	counter = 0;
	for(itc = constraints_.begin(); itc != constraints_.end(); ++itc) {
		dual_prices_[itc->first] = duals[counter++];
	}
}

ostream& operator<<(ostream& os, const CoinModelWrapper& cmw) {
	os << "min ";
	map<string, bool>::const_iterator itvars;
	map<string, double>::const_iterator itcoefs;
	map<string, double>::const_iterator itsoln;

	//objective
	for(itvars = cmw.variables_.begin(); itvars != cmw.variables_.end(); ++itvars) {
		itcoefs = cmw.variable_objective_coefs_.find(itvars->first);
		if(itcoefs != cmw.variable_objective_coefs_.end()) {
			if(itcoefs->second > 0) {
				os << "+";
			}
			os << itcoefs->second << "*" << itvars->first << " ";
		}
	}
	os << endl;

	//subject to
	map<string, bool>::const_iterator itconstr;
	map<string, map<string, double> *>::const_iterator itconstvar;
	for(itconstr = cmw.constraints_.begin(); itconstr != cmw.constraints_.end(); ++itconstr) {
		os << itconstr->first << ": ";
		itcoefs = cmw.constraint_lower_bounds_.find(itconstr->first);
		if(itcoefs != cmw.constraint_lower_bounds_.end()) {
			os << itcoefs->second << " <= ";
		}
		itconstvar = cmw.coinstraint_variables_.find(itconstr->first);
		for(itvars = cmw.variables_.begin(); itvars != cmw.variables_.end(); ++itvars) {
			itcoefs = itconstvar->second->find(itvars->first);
			if(itcoefs != itconstvar->second->end()) {
				if(itcoefs->second > 0) {
					os << "+";
				}
				os << itcoefs->second << "*" << itvars->first << " ";
			}
		}
		itcoefs = cmw.constraint_upper_bounds_.find(itconstr->first);
		if(itcoefs != cmw.constraint_upper_bounds_.end()) {
			os << " <= " << itcoefs->second;
		}
		os << endl;
	}

	//variable bounds
	for(itvars = cmw.variables_.begin(); itvars != cmw.variables_.end(); ++itvars) {
		itcoefs = cmw.variable_lower_bounds_.find(itvars->first);
		if(itcoefs != cmw.variable_lower_bounds_.end()) {
			os << itcoefs->second << " <= ";
		}
		os << itvars->first;
		itcoefs = cmw.variable_upper_bounds_.find(itvars->first);
		if(itcoefs != cmw.variable_upper_bounds_.end()) {
			os << " <= " << itcoefs->second;
		}
		os << endl;
	}
	os << endl;

	//solution
	for(itsoln = cmw.variable_solutions_.begin(); itsoln != cmw.variable_solutions_.end(); ++itsoln) {
		os << itsoln->first << " = " << itsoln->second << endl;
	}
	os << endl;
	
	return os;
}
