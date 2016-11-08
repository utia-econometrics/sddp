cmake_minimum_required(VERSION 2.8.9)
project(sddp)
 
#Bring the headers, such as Student.h into the project

include_directories(source "/usr/local/boost_1_57_0")
 
#Can manually add the sources using the set command as follows:
#set(SOURCES src/mainapp.cpp src/Student.cpp)
 
#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES "source/*.cpp")

#For the static library:
set ( PROJECT_LINK_LIBS boost.a )
#link_directories( ~/exploringBB/extras/cmake/studentlib_static/build )

add_executable(sddp ${SOURCES})

target_link_libraries(sddp ${PROJECT_LINK_LIBS} )

