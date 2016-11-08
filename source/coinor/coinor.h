//force the CLP,CGC and CGL etc. to build
#define CLP_BUILD 1
#define CBC_BUILD 1
#define CGL_BUILD 1
#define OSI_BUILD 1
#define COINUTILS_BUILD 1

//select the default solver
//#define CBC_DEFAULT_SOLVER "clp"

/* now defined by cbc
//some extern stuff to make COIN-OR link (ugly)
int CbcOrClpRead_mode = 1;
FILE * CbcOrClpReadCommand = stdin;
extern int CbcOrClpEnvironmentIndex;
*/