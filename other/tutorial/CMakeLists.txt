include_directories( ../../Population )
SET(EXECUTABLE_OUTPUT_PATH ${POPULATION_LIB_DIR})

add_executable(tutorial_firststep firststep.cpp)
target_link_libraries (tutorial_firststep ${POPULATION_LIBRARY} ${POPULATION_LIB_NAME})
link_directories (${POPULATION_LIB_DIR})

add_executable(tutorial_basic_definition basic_definition.cpp)
target_link_libraries (tutorial_basic_definition ${POPULATION_LIBRARY} ${POPULATION_LIB_NAME})

add_executable(tutorial_loadsave loadsave.cpp)
target_link_libraries (tutorial_loadsave ${POPULATION_LIBRARY} ${POPULATION_LIB_NAME})

add_executable(tutorial_fastprototyping3d fastprototyping3d.cpp)
target_link_libraries (tutorial_fastprototyping3d ${POPULATION_LIBRARY} ${POPULATION_LIB_NAME})

add_executable(tutorial_segmentation_threshold segmentation.cpp)
target_link_libraries (tutorial_segmentation_threshold ${POPULATION_LIBRARY} ${POPULATION_LIB_NAME})


add_executable(tutorial_segmentation_watershed segmentation2.cpp)
target_link_libraries (tutorial_segmentation_watershed ${POPULATION_LIBRARY} ${POPULATION_LIB_NAME})


add_executable(tutorial_segmentation_swamping segmentation3.cpp)
target_link_libraries (tutorial_segmentation_swamping ${POPULATION_LIBRARY} ${POPULATION_LIB_NAME})


add_executable(tutorial_visualization_2d visualization2d.cpp)
target_link_libraries (tutorial_visualization_2d ${POPULATION_LIBRARY} ${POPULATION_LIB_NAME})


add_executable(tutorial_visualization_3d visualization3d.cpp)
target_link_libraries (tutorial_visualization_3d ${POPULATION_DEP_LIBRARIES} population)

add_executable(tutorial_template_programming template_programming.cpp)
target_link_libraries (tutorial_template_programming ${POPULATION_DEP_LIBRARIES} population)

add_executable(tutorial_iterator_matrix iterator_matrix.cpp)
target_link_libraries (tutorial_iterator_matrix ${POPULATION_DEP_LIBRARIES} population)



add_executable(PatternRecognitionMachineLearningBishopChapter1 PatternRecognitionMachineLearningBishopChapter1.cpp)
target_link_libraries (PatternRecognitionMachineLearningBishopChapter1 ${POPULATION_DEP_LIBRARIES} population)
