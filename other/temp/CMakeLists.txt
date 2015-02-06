include_directories( ../../Population )
SET(EXECUTABLE_OUTPUT_PATH ${POPULATION_LIB_DIR})
add_executable(temp_neuralnetworkmatrix main.cpp neuralnetworkmatrix.h)
target_link_libraries (temp_neuralnetworkmatrix ${POPULATION_LIBRARY} ${POPULATION_LIB_NAME})
link_directories (${POPULATION_LIB_DIR})

