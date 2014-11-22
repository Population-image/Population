include_directories( ../../Population )
SET(EXECUTABLE_OUTPUT_PATH ${POPULATION_LIB_DIR})
add_executable(treillisnumeric main.cpp treillisnumeric.cpp treillisnumeric.h neuralnetworkmatrix.h Cluster.cpp Cluster.h)
target_link_libraries (treillisnumeric ${POPULATION_LIB_NAME})
link_directories (${POPULATION_LIB_DIR})

