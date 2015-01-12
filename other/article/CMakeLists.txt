include_directories( ../../Population )
SET(EXECUTABLE_OUTPUT_PATH ${POPULATION_LIB_DIR})


add_executable(article_elementary_decomposition elementarydecomposition.cpp)
target_link_libraries (article_elementary_decomposition ${POPULATION_LIB_NAME})
link_directories (${POPULATION_LIB_DIR})

add_executable(article_germgrain germgrain.cpp)
target_link_libraries (article_germgrain ${POPULATION_LIB_NAME})
link_directories (${POPULATION_LIB_DIR})
