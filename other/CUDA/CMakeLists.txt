cmake_minimum_required(VERSION 2.6)


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


link_directories (${POPULATION_LIB_DIR})
include_directories(${POPULATION_INCLUDE_DIRS})
CUDA_ADD_LIBRARY(TestLib STATIC popcuda.h popcuda.cc c_neural_net.h c_neural_net.cu)
CUDA_COMPILE(CUDA_TEST microtime.h cuda_test.h cuda_test.cu)
ADD_EXECUTABLE(CUDA_test CUDA_test.cpp neural_net_test.h neural_net_test.cc ${CUDA_TEST} ${POPULATION_SOURCES})
ADD_DEPENDENCIES(CUDA_test TestLib)
CUDA_ADD_CUBLAS_TO_TARGET(CUDA_test)
TARGET_LINK_LIBRARIES(CUDA_test TestLib ${POPULATION_LIBRARY})

