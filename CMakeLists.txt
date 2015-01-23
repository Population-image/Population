cmake_minimum_required(VERSION 2.6)

#Déclaration du projet
project(population)

set(POP_MAJOR_VERSION 5)
set(POP_MINOR_VERSION 0)
set(POP_PATCH_VERSION 0)
set(POP_VERSION ${POP_MAJOR_VERSION}.${POP_MINOR_VERSION}.${POP_PATCH_VERSION})

set(CMAKE_CONFIGURATION_TYPES "Release;Debug" CACHE STRING "Configs" FORCE)
#if(DEFINED CMAKE_BUILD_TYPE AND CMAKE_VERSION VERSION_GREATER "2.8")
#  set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES} )
#endif()
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_VERBOSE_MAKEFILE 1)


# ----------------------------------------------------------------------------
# Population cmake options
# ----------------------------------------------------------------------------
include(cmake/popUtils.cmake)

# Optional 3rd party components
# ===================================================
POP_OPTION(WITH_CIMG         "Include CIMG"                      ON   IF (NOT ANDROID))
POP_OPTION(WITH_OPENGL       "Include OpenGl"                      ON   IF (NOT ANDROID))
POP_OPTION(WITH_OPENMP       "Include OpenMP for parallization"   OFF   IF (NOT ANDROID))
POP_OPTION(WITH_FFMPEG       "Include FFMPEG support"                      OFF   IF (NOT ANDROID AND NOT IOS))
POP_OPTION(WITH_VLC          "Include VLC support"                      OFF   IF (NOT ANDROID AND NOT IOS))
POP_OPTION(WITH_QT           "Include QT"                      OFF   IF (NOT ANDROID))
POP_OPTION(WITH_CUDA         "Include Cuda"   OFF   IF (NOT ANDROID))

# Population build components
# ===================================================
POP_OPTION(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON )
POP_OPTION(BUILD_EXECUTABLE  "Build executable" ON )
POP_OPTION(BUILD_PYTHON_SWIG "Build SWIG PYTHON" OFF )
POP_OPTION(BUILD_TUTORIAL    "Build tutorial" ON )
POP_OPTION(BUILD_ARTICLE     "Build article code" ON )
POP_OPTION(BUILD_TEMP        "Build experimental code" OFF )
POP_OPTION(BUILD_TEST        "Test" OFF )
POP_OPTION(BUILD_CUDA        "Cuda test" OFF )
POP_OPTION(BUILD_EXPORT      "export library" OFF )
POP_OPTION(CMAKE_VERBOSE     "Verbose mode" OFF )


# Offer the user the choice of overriding the installation directories
set(POPULATION_LIB_DIR ${PROJECT_BINARY_DIR} CACHE PATH "Installation directory for libraries")
set(POPULATION_BIN_DIR ${PROJECT_BINARY_DIR} CACHE PATH "Installation directory for executables")
set(POPULATION_CMAKE_DIR  CMake CACHE PATH  "Installation directory for CMake files")
set(POPULATION_LIB_NAME population CACHE PATH  "Installation lib name")
set(POPULATION_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include;${PROJECT_BINARY_DIR}/include" CACHE PATH "Include directory")





# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var POPULATION_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_CURRENT_SOURCE_DIR}/${${var}}")
  endif()
endforeach()


#source file
file(
        GLOB_RECURSE
        source_file
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/include/*.h
        ${PROJECT_SOURCE_DIR}/include/*.hpp
)
set(POPULATION_SOURCES "${source_file}" CACHE STRING "population library")


if(CMAKE_VERBOSE)
  set(CMAKE_VERBOSE_MAKEFILE 1)
endif()

# ----------------------------------------------------------------------------
# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if(CMAKE_GENERATOR MATCHES "Makefiles|Ninja" AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()


# ----------------------------------------------------------------------------
#  Detect 3rd-party libraries
# ----------------------------------------------------------------------------
if(WITH_CIMG)
    set(WITH_X11 true)
    set(WITH_THREAD true)
    set(HAVE_CIMG true)
endif()
include(cmake/FindFFMPEG.cmake)
include(cmake/FindVLC.cmake)
include(cmake/FindOPENGL.cmake)
include(cmake/FindOPENMP.cmake)
include(cmake/FindTHREAD.cmake)
include(cmake/FindX11.cmake)
include(cmake/FindCUDA.cmake)

if(WITH_QT)
    set(HAVE_QT true)
endif(WITH_QT)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUNICODE=1")


# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if(CMAKE_GENERATOR MATCHES "Makefiles|Ninja" AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

# platform-specific config file
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/popconfig.h.in" "${PROJECT_BINARY_DIR}/include/popconfig.h")
include_directories(${POPULATION_INCLUDE_DIRS})
if(POPULATION_LIBRARY)
    list(REMOVE_DUPLICATES POPULATION_LIBRARY)
endif()

if(BUILD_SHARED_LIBS OR BUILD_TUTORIAL OR BUILD_ARTICLE OR BUILD_TEMP )
    add_library(${POPULATION_LIB_NAME}  SHARED ${POPULATION_SOURCES})
    target_link_libraries(${POPULATION_LIB_NAME} ${POPULATION_LIBRARY})
if(BUILD_EXPORT)
    file(COPY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${PROJECT_BINARY_DIR}/include)
    file(COPY ${PROJECT_SOURCE_DIR}/image/ DESTINATION ${PROJECT_BINARY_DIR}/image)
    file(COPY ${PROJECT_SOURCE_DIR}/file/ DESTINATION ${PROJECT_BINARY_DIR}/file)
    install(TARGETS ${POPULATION_LIB_NAME}
  # IMPORTANT: Add the population library to the "export-set"
  EXPORT PopulationTargets
  RUNTIME DESTINATION "${POPULATION_LIB_DIR}" COMPONENT bin
  LIBRARY DESTINATION "${POPULATION_LIB_DIR}" COMPONENT shlib
  PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/include/"
    COMPONENT dev)
endif(BUILD_EXPORT)
endif(BUILD_SHARED_LIBS OR BUILD_TUTORIAL OR BUILD_ARTICLE OR BUILD_TEMP )


if(BUILD_EXECUTABLE)
    add_executable(test_main main.cpp ${POPULATION_SOURCES})
    target_link_libraries (test_main ${POPULATION_LIBRARY})
    add_test(test1 test_main)
 #   find_package( OpenCV REQUIRED )
 #   include_directories( ${OpenCV_INCLUDE_DIRS} )
 #   target_link_libraries( test_main ${OpenCV_LIBS} )
endif(BUILD_EXECUTABLE)

if(BUILD_PYTHON_SWIG)
    add_subdirectory(other/python/)
endif(BUILD_PYTHON_SWIG)

if(BUILD_TUTORIAL)
    add_subdirectory(other/tutorial/)
endif(BUILD_TUTORIAL)

if(BUILD_ARTICLE)
    add_subdirectory(other/article/)
endif(BUILD_ARTICLE)

if(BUILD_TEMP)
    add_subdirectory(other/temp/)
endif(BUILD_TEMP)

if(BUILD_TEST)
    add_subdirectory(other/test/)
endif(BUILD_TEST)

if(BUILD_CUDA)
    add_subdirectory(other/CUDA/)
endif(BUILD_CUDA)


if(BUILD_EXPORT)
    # Add all targets to the build-tree export set
    export(TARGETS ${POPULATION_LIB_NAME}  FILE "${PROJECT_BINARY_DIR}/PopulationTargets.cmake")

    # Export the package for use from the build-tree
    # (this registers the build-tree with a global CMake-registry)
    export(PACKAGE Population)

    # Create the PopulationConfig.cmake and PopulationConfigVersion files
    file(RELATIVE_PATH REL_INCLUDE_DIR "${POPULATION_CMAKE_DIR}" "${PROJECT_SOURCE_DIR}")
    # ... for the build tree
    set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/include")
    configure_file(cmake/PopulationConfig.cmake.in "${PROJECT_BINARY_DIR}/PopulationConfig.cmake" @ONLY)
    # ... for the install tree
    set(CONF_INCLUDE_DIRS "\${POPULATION_CMAKE_DIR}/${REL_INCLUDE_DIR}")
    configure_file(cmake/PopulationConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PopulationConfig.cmake" @ONLY)
    # ... for both
    configure_file(cmake/PopulationConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/PopulationConfigVersion.cmake" @ONLY)

    # Install the PopulationConfig.cmake and PopulationConfigVersion.cmake
    install(FILES
        "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/PopulationConfig.cmake"
        "${PROJECT_BINARY_DIR}/PopulationConfigVersion.cmake"
        DESTINATION "${POPULATION_CMAKE_DIR}" COMPONENT dev)

    # Install the export set for use with the install-tree
    install(EXPORT PopulationTargets DESTINATION
        "${POPULATION_CMAKE_DIR}" COMPONENT dev)

endif(BUILD_EXPORT)
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
    set(POPULATION_LIBRARIES "${POPULATION_LIBRARY}" PARENT_SCOPE)
    set(POPULATION_INCLUDE_DIRS "${POPULATION_INCLUDE_DIRS}" PARENT_SCOPE)
endif()






