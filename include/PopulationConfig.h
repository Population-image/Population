#ifndef POPULATIONCONFIG_H
#define POPULATIONCONFIG_H
#include <assert.h>
#include <string>
#if !defined(HAVE_QMAKE)
#include"popconfig.h"
#endif

#ifndef POP_PROJECT_SOURCE_DIR
#define POP_PROJECT_SOURCE_DIR ""
#endif

namespace pop{
// Disable silly warnings on some Microsoft VC++ compilers.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#endif

#if defined(unix)        || defined(__unix)      || defined(__unix__) \
 || defined(linux)       || defined(__linux)     || defined(__linux__) \
 || defined(sun)         || defined(__sun) \
 || defined(BSD)         || defined(__OpenBSD__) || defined(__NetBSD__) \
 || defined(__FreeBSD__) || defined __DragonFly__ \
 || defined(sgi)         || defined(__sgi) \
 || defined(__MACOSX__)  || defined(__APPLE__) \
 || defined(__CYGWIN__)
#define Pop_OS 1
#elif defined(_MSC_VER) || defined(WIN32)  || defined(_WIN32) || defined(__WIN32__) \
   || defined(WIN64)    || defined(_WIN64) || defined(__WIN64__)
#define Pop_OS 2
#define UNICODE 1
#else
#define Pop_OS 0
#endif


#ifdef HAVE_OPENMP
#include "omp.h"
#endif

#if Pop_OS==2
    #define POP_EXPORTS __declspec(dllexport)
#else
    #define POP_EXPORTS
#endif

#ifdef HAVE_DEBUG
#define POP_DbgAssert(expr) assert(expr)
#else
#define POP_DbgAssert(expr)
#endif


#ifdef HAVE_DEBUG
#   define POP_DbgAssertMessage(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (false)
#else
#   define POP_DbgAssertMessage(condition, message) do { (void)(condition);(void) (message); } while (0)
#endif

template<typename Function, typename F>
struct FunctionTypeTraitsSubstituteF
{
    typedef F Result;
};



template<typename Function, int DIM>
struct FunctionTypeTraitsSubstituteDIM
{
    typedef Function Result;
};
template<
        typename Function1,
        typename Function2
        >
void FunctionAssert(const  Function1 & ,const  Function2 &  ,std::string )
{
}
}
#endif
