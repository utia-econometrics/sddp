# SDDP C++ project

**In order to compile, you need to install following dependencies:**

- Boost C++ library http://www.boost.org/, version at least 1.61, if installed as Debian package then no special actions should be needed. The package may be installed by `apt-get install libboost-all-dev` 
- COIN-OR CBC package https://projects.coin-or.org/BuildTools/wiki/downloadUnix, after compiling check that the paths in CMakeLists.txt are valid
- IBM ILOG Cplex Academic Edition https://developer.ibm.com/academic/, after installation check paths for cplex and concert files in CMakeLists.txt
- Armadillo matrix library http://arma.sourceforge.net/, if installed as package (apt-get install libarmadillo-dev) no special actions should be needed

For compiling debug version of the project, run

`cmake -DCMAKE_BUILD_TYPE=Debug ..`

Example run of the program to solve asset model with data file data.txt
 sddp asset data.txt
