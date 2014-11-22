include_directories( ../../Population )
SET(EXECUTABLE_OUTPUT_PATH ${POPULATION_LIB_DIR})
add_executable(firststep firststep.cpp)
target_link_libraries (firststep ${POP_LINKER_LIBS} population)
link_directories (${POPULATION_LIB_DIR})

add_executable(tutorial_segmentation_threshold segmentation.cpp)
target_link_libraries (tutorial_segmentation_threshold ${POPULATION_DEP_LIBRARIES} population)


add_executable(tutorial_segmentation_watershed segmentation2.cpp)
target_link_libraries (tutorial_segmentation_watershed ${POPULATION_DEP_LIBRARIES} population)


add_executable(tutorial_segmentation_swamping segmentation3.cpp)
target_link_libraries (tutorial_segmentation_swamping ${POPULATION_DEP_LIBRARIES} population)


add_executable(tutorial_visualization_2d visualization2d.cpp)
target_link_libraries (tutorial_visualization_2d ${POPULATION_DEP_LIBRARIES} population)


add_executable(tutorial_visualization_3d visualization3d.cpp)
target_link_libraries (tutorial_visualization_3d ${POPULATION_DEP_LIBRARIES} population)
