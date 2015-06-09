if(NOT WITH_THREAD)
          message(STATUS "THREAD NOT")
  	return()
endif()
if(UNIX)
  	find_package (Threads REQUIRED)
        if(CMAKE_USE_PTHREADS_INIT)
                set(POPULATION_LIBRARY ${POPULATION_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} )
    		set(HAVE_THREAD YES)
                message(STATUS "THREAD FOUND" ${POPULATION_LIBRARY})
                message(STATUS ${CMAKE_THREAD_LIBS_INIT})
        else()
                 message(STATUS "THREAD NOT FOUND")
        endif(CMAKE_USE_PTHREADS_INIT)
endif(UNIX)

if(WIN32)
    set(HAVE_THREAD YES)
endif(WIN32)
