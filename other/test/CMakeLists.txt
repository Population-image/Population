cmake_minimum_required(VERSION 2.6)
project( test_main )
link_directories (${POPULATION_LIB_DIR})
include_directories(${POPULATION_INCLUDE_DIRS})
add_executable( runtest runtest.cpp poptest.h ${POPULATION_SOURCES})
target_link_libraries(runtest ${POPULATION_LIBRARY})


add_executable( adaboost adaboost.cpp  ${POPULATION_SOURCES})
target_link_libraries(adaboost ${POPULATION_LIBRARY})



