if(NOT WITH_VLC)
  message(STATUS "VLC NOT")
  return()
endif()



# CMake module to search for LIBVLC (VLC library)
# Authors: Rohit Yadav <rohityadav89@gmail.com>
# Harald Sitter <apachelogger@ubuntu.com>
#
# If it's found it sets FIND_VLC to TRUE
# and following variables are set:
# LIBVLC_INCLUDE_DIR
# LIBVLC_LIBRARY
# LIBVLC_VERSION

if(NOT LIBVLC_MIN_VERSION)
    set(LIBVLC_MIN_VERSION "0.0")
endif(NOT LIBVLC_MIN_VERSION)

# find_path and find_library normally search standard locations
# before the specified paths. To search non-standard paths first,
# FIND_* is invoked first with specified paths and NO_DEFAULT_PATH
# and then again with no specified paths to search the default
# locations. When an earlier FIND_* succeeds, subsequent FIND_*s
# searching for the same item do nothing.

if (NOT WIN32)
    find_package(PkgConfig)
    pkg_check_modules(PC_LIBVLC libvlc)
    set(LIBVLC_DEFINITIONS ${PC_LIBVLC_CFLAGS_OTHER})
endif (NOT WIN32)

#Put here path to custom location
#example: /home/user/vlc/include etc..
find_path(LIBVLC_INCLUDE_DIR vlc/vlc.h
HINTS "$ENV{LIBVLC_INCLUDE_PATH}"
PATHS
    "$ENV{LIB_DIR}/include"
    "$ENV{LIB_DIR}/include/vlc"
    "/usr/include"
    "/usr/include/vlc"
    "/usr/local/include"
    "/usr/local/include/vlc"
    #mingw
    c:/msys/local/include
    "C:/Program Files (x86)/VideoLAN/VLC/sdk/include"
)

find_path(LIBVLC_INCLUDE_DIR PATHS "${CMAKE_INCLUDE_PATH}/vlc" NAMES vlc.h
        HINTS ${PC_LIBVLC_INCLUDEDIR} ${PC_LIBVLC_INCLUDE_DIRS})
#Put here path to custom location
#example: /home/user/vlc/lib etc..
find_library(LIBVLC_LIBRARY NAMES vlc libvlc
HINTS "$ENV{LIBVLC_LIBRARY_PATH}" ${PC_LIBVLC_LIBDIR} ${PC_LIBVLC_LIBRARY_DIRS}
PATHS
    "$ENV{LIB_DIR}/lib"
    #mingw
    c:/msys/local/lib
    "C:/Program Files (x86)/VideoLAN/VLC/"
)
find_library(LIBVLC_LIBRARY NAMES vlc libvlc)
find_library(LIBVLCCORE_LIBRARY NAMES vlccore libvlccore
HINTS "$ENV{LIBVLC_LIBRARY_PATH}" ${PC_LIBVLC_LIBDIR} ${PC_LIBVLC_LIBRARY_DIRS}
PATHS
    "$ENV{LIB_DIR}/lib"
    #mingw
    c:/msys/local/lib
    "C:/Program Files (x86)/VideoLAN/VLC/"
)


find_library(LIBVLCCORE_LIBRARY NAMES vlccore libvlccore)

set(LIBVLC_VERSION ${PC_LIBVLC_VERSION})
if (NOT LIBVLC_VERSION)
# TODO: implement means to detect version on windows (vlc --version && regex? ... ultimately we would get it from a header though...)
endif (NOT LIBVLC_VERSION)
if (LIBVLC_INCLUDE_DIR AND LIBVLC_LIBRARY AND LIBVLCCORE_LIBRARY)
set(FIND_VLC TRUE)
endif (LIBVLC_INCLUDE_DIR AND LIBVLC_LIBRARY AND LIBVLCCORE_LIBRARY)

if (LIBVLC_VERSION STRLESS "${LIBVLC_MIN_VERSION}")
    message(WARNING "LibVLC version not found: version searched: ${LIBVLC_MIN_VERSION}, found ${LIBVLC_VERSION}\nUnless you are on Windows this is bound to fail.")
# TODO: only activate once version detection can be garunteed (which is currently not the case on windows)
# set(FIND_VLC FALSE)
endif (LIBVLC_VERSION STRLESS "${LIBVLC_MIN_VERSION}")

if (FIND_VLC)
    message(STATUS "Found LibVLC include-dir path: ${LIBVLC_INCLUDE_DIR}")
    message(STATUS "Found LibVLC library path:${LIBVLC_LIBRARY}")
    message(STATUS "Found LibVLCcore library path:${LIBVLCCORE_LIBRARY}")
    message(STATUS "Found LibVLC version: ${LIBVLC_VERSION} (searched for: ${LIBVLC_MIN_VERSION})")

    if (NOT LIBVLC_FIND_QUIETLY)
        set(HAVE_VLC true)
        include_directories(${LIBVLC_INCLUDE_DIR})
        set(POPULATION_LIBRARY ${POPULATION_LIBRARY} ${LIBVLC_LIBRARY} ${LIBVLCCORE_LIBRARY})
        set(POPULATION_INCLUDE_DIRS ${POPULATION_INCLUDE_DIRS} ${LIBVLC_INCLUDE_DIR})
    endif (NOT LIBVLC_FIND_QUIETLY)
else (FIND_VLC)
    message(status "Could not find LibVLC (in Debian/Ubuntu you need libvlc-dev and libvlccore-dev packages)")
    if (LIBVLC_FIND_REQUIRED)
        message(status "VLC NOT FOUND")
    endif (LIBVLC_FIND_REQUIRED)
endif (FIND_VLC)
