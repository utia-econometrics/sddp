#include "GamsWrapper.h"
#include "ExeRunner.h"


const string GamsWrapper::GAMS_TMP_OUT = "gams.out";

///when constructing, please pass the path to gams executable
GamsWrapper::GamsWrapper(const string &gamsPath)
{
	gamsPath_ = gamsPath;
}

GamsWrapper::~GamsWrapper(void)
{
}

///pass the script name to be executed
string GamsWrapper::RunGams(const string &scriptName)
{
	//prepare the command line params
	string params;
	params = " \"" + scriptName + "\"";

	//
	// tell gams to write to stdout
	//
	params += " LO=3 LL=1 o=";
	params += GAMS_TMP_OUT;
	
	//omit the stdout
	ExeRunner::RunExe(gamsPath_, params);

	//read the output file
	ifstream infile;
	infile.open(GAMS_TMP_OUT.c_str(), ios::in);
	if(!infile.is_open()) {
		throw GamsWrapperException("Failed opening the output file");
	}

	string line;
	stringstream str;
	while(getline(infile,line)) {
		str << line << endl;
	}
	return str.str();
}