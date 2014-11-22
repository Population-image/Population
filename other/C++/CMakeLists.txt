cmake_minimum_required(VERSION 2.6)
project( test_main )
link_directories (${POP_LIBRARY_OUTPUT_DIRECTORY})
include_directories( ${POP_INCLUDE_DIRS} )
add_executable( test_main main.cpp )
target_link_libraries(test_main ${POP_LINKER_LIBS})






