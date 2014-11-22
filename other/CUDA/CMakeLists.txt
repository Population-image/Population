cmake_minimum_required(VERSION 2.6)

#project( CUDA_test )
#link_directories (${POPULATION_LIB_DIR})
#include_directories(${POPULATION_INCLUDE_DIRS})
#add_executable( CUDA_test CUDA_test.cpp ${POPULATION_SOURCES})
#target_link_libraries(CUDA_test ${POPULATION_LIBRARY})

############################################################
#													 	   #
# CMake will generate a Makefile that compiles the program #
# and links it to libcudart.so. The best way to be able to #
# run the program even if CUDA is not installed on the     #
# machine (e.g., because there is no CUDA capable device)  #
# is to copy the libcudart.so file in the binary directory #
# and then add it to the LD_LIBRARY_PATH:                  #
#  $ LD_LIBRARY_PATH=. ./cuda_test                         #
#  You don't have a CUDA device.						   #
#													 	   #
############################################################


CUDA_ADD_LIBRARY(TestLib STATIC popcuda.h popcuda.cc)
CUDA_COMPILE(CUDA_TEST microtime.h cuda_test.h cuda_test.cu)
ADD_EXECUTABLE(CUDA_test CUDA_test.cpp ${CUDA_TEST})
ADD_DEPENDENCIES(CUDA_test TestLib)
TARGET_LINK_LIBRARIES(CUDA_test TestLib)