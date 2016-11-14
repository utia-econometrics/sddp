cmake_minimum_required(VERSION 2.8.9)
project(sddp)

#config - directories
set( CPLEX_HEADER_DIR "/opt/ibm/ILOG/CPLEX_Studio1263/cplex/include" )
set( CONCERT_HEADER_DIR "/opt/ibm/ILOG/CPLEX_Studio1263/concert/include" )
set( COIN_HEADER_DIR "/usr/local/include/coin" )
set( CPLEX_LIB_DIR "/opt/ibm/ILOG/CPLEX_Studio1263/cplex/lib/x86-64_linux/static_pic" )
set( CPLEX_BIN_DIR "/opt/ibm/ILOG/CPLEX_Studio1263/cplex/bin/x86-64_linux" )
set( CONCERT_LIB_DIR "/opt/ibm/ILOG/CPLEX_Studio1263/concert/lib/x86-64_linux/static_pic" )
set( COIN_LIB_DIR "/usr/local/lib/coin" )
set( CPLEX_VER "1263" )

#config - optimization
set( CMAKE_BUILD_TYPE "Release" )

#COIN-OR headers, put your directory
include_directories(source ${COIN_HEADER_DIR} )

#CPLEX - force it to use STD headers like <iostream> instead of <iostream.h>
add_definitions( -DIL_STD )
#CPLEX - headers, put your directory
include_directories(source ${CPLEX_HEADER_DIR} )
include_directories(source ${CONCERT_HEADER_DIR} )
 
#Can manually add the sources using the set command as follows:
#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "source/*.cpp" "source/alglib/*.cpp" "source/rngstream/*.cpp" )

#CPLEX - libraries to link
find_library(ILOCPLEX libilocplex.a ${CPLEX_LIB_DIR} )
find_library(CPLEX libcplex.a ${CPLEX_LIB_DIR} )
find_library(BINCPLEX libcplex${CPLEX_VER}.so ${CPLEX_BIN_DIR} )
find_library(CONCERT libconcert.a ${CONCERT_LIB_DIR} )
#COIN-OR libraries
find_library(COINCBC libCbc.so ${COIN_LIB_DIR} )
find_library(COINCBCSOLVER libCbcSolver.so ${COIN_LIB_DIR} )
find_library(COINCLP libClp.so ${COIN_LIB_DIR} )
find_library(COINCLPSOLVER libClpSolver.so ${COIN_LIB_DIR} )
find_library(COINCGL libCgl.so ${COIN_LIB_DIR} )
find_library(COINUTILS libCoinUtils.so ${COIN_LIB_DIR} )
find_library(COINOSI libOsi.so ${COIN_LIB_DIR} )
find_library(COINOSICLP libOsiClp.so ${COIN_LIB_DIR} )
find_library(COINOSICBC libOsiCbc.so ${COIN_LIB_DIR} )

#Boost libraries
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

#Armadillo library
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

#Threads for boost and CPLEX
find_package (Threads REQUIRED)

#Build our executable
add_executable(sddp ${SOURCES})

#Link with specified libraries
target_link_libraries(sddp ${ILOCPLEX} )
target_link_libraries(sddp ${CPLEX} )
target_link_libraries(sddp ${BINCPLEX} )
target_link_libraries(sddp ${CONCERT} )
target_link_libraries(sddp ${COINCBC} )
target_link_libraries(sddp ${COINCBCSOLVER} )
target_link_libraries(sddp ${COINCLP} )
target_link_libraries(sddp ${COINCLPSOLVER} )
target_link_libraries(sddp ${COINCGL} )
target_link_libraries(sddp ${COINUTILS} )
target_link_libraries(sddp ${COINOSI} )
target_link_libraries(sddp ${COINOSICLP} )
target_link_libraries(sddp ${COINOSICBC} )
target_link_libraries(sddp ${Boost_LIBRARIES})
target_link_libraries(sddp ${ARMADILLO_LIBRARIES})
target_link_libraries(sddp ${CMAKE_THREAD_LIBS_INIT})
#required for CPLEX
target_link_libraries(sddp ${CMAKE_DL_LIBS} )
