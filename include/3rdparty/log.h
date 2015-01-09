#ifndef __LOG_H__
#define __LOG_H__

#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <stdio.h>

//! redefinition de l'appel du nom de fonction car depend du type de compilateur
#if defined (__FUNCSIG__)
#   define FUNCTION_SIG __FUNCSIG__
#elif defined(__PRETTY_FUNCTION__)
#   define FUNCTION_SIG __PRETTY_FUNCTION__
#else
#   define FUNCTION_SIG __FUNCTION__
#endif

using namespace std;

inline std::string NowTime();

enum TLogLevel {logERROR, logWARNING, logINFO, logDEBUG, logDEBUG1, logDEBUG2, logDEBUG3, logDEBUG4};

template<typename T> std::string getClassName(T const &obj)
{
    return typeid(obj).name();
}

template <typename T>
class Log
{
public:
    Log();
    virtual ~Log();
    std::ostringstream& Get(TLogLevel level = logINFO, std::string className = "", std::string funcName = "");
public:
    static TLogLevel& ReportingLevel();
    static std::string ToString(TLogLevel level);
    static TLogLevel FromString(const std::string& level);
protected:
    std::ostringstream os;
private:
    Log(const Log&);
    Log& operator =(const Log&);
};

template <typename T>
Log<T>::Log()
{
}

//! Ligne de texte affiché dans le fichier
template <typename T>
std::ostringstream& Log<T>::Get(TLogLevel level, std::string className, std::string funcName)
{
    os << "- " << NowTime();
    os << " " << ToString(level) << ": ";
    if (funcName != "") os << "from " << funcName << " ";
    if (className != "") os <<  "in " << className << ": ";
    os << std::string(level > logDEBUG ? level - logDEBUG : 0, '\t');
    return os;
}

template <typename T>
Log<T>::~Log()
{
    os << std::endl;
    T::Output(os.str());
}

template <typename T>
TLogLevel& Log<T>::ReportingLevel()
{
    static TLogLevel reportingLevel = logDEBUG4;
    return reportingLevel;
}

template <typename T>
std::string Log<T>::ToString(TLogLevel level)
{
    static const char* const buffer[] = {"ERROR", "WARNING", "INFO", "DEBUG", "DEBUG1", "DEBUG2", "DEBUG3", "DEBUG4"};
    return buffer[level];
}

template <typename T>
TLogLevel Log<T>::FromString(const std::string& level)
{
    if (level == "DEBUG4")
        return logDEBUG4;
    if (level == "DEBUG3")
        return logDEBUG3;
    if (level == "DEBUG2")
        return logDEBUG2;
    if (level == "DEBUG1")
        return logDEBUG1;
    if (level == "DEBUG")
        return logDEBUG;
    if (level == "INFO")
        return logINFO;
    if (level == "WARNING")
        return logWARNING;
    if (level == "ERROR")
        return logERROR;
    Log<T>().Get(logWARNING) << "Unknown logging level '" << level << "'. Using INFO level as default.";
    return logINFO;
}

//! Classe du fichier de sortie a instancier dans le main
class Output2FILE
{
public:
    static FILE*& Stream();
    static void Archivage();
    static void Output(const std::string& msg);
};

//! méthode d'affectation du flux d'informations au fichier de sortie
inline FILE*& Output2FILE::Stream()
{
    static FILE* pStream = stderr;
    return pStream;
}

//! méthode d'archivage du log
inline void Output2FILE::Archivage()
{
//    std::cout << "_tmpfname : " << Output2FILE::Stream()->_tmpfname << "FIN" << std::endl;
//    std::cout << "_base : " << Output2FILE::Stream()->_base << "FIN" << std::endl;
//    std::cout << "_ptr : " << Output2FILE::Stream()->_ptr << "FIN" << std::endl;
}


inline void Output2FILE::Output(const std::string& msg)
{
    FILE* pStream = Stream();
    if (!pStream)
        return;
    fprintf(pStream, "%s", msg.c_str());
    fflush(pStream);
}

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#   if defined (BUILDING_FILELOG_DLL)
#       define FILELOG_DECLSPEC   __declspec (dllexport)
#   elif defined (USING_FILELOG_DLL)
#       define FILELOG_DECLSPEC   __declspec (dllimport)
#   else
#       define FILELOG_DECLSPEC
#   endif // BUILDING_DBSIMPLE_DLL
#else
#   define FILELOG_DECLSPEC
#endif // _WIN32

class FILELOG_DECLSPEC FILELog : public Log<Output2FILE> {};
//typedef Log<Output2FILE> FILELog;

#ifndef FILELOG_MAX_LEVEL
#define FILELOG_MAX_LEVEL logDEBUG4
#endif

//! Macros a appeler dans le code pour ajouter des lignes au fichier log
//! Différentes macros appelées selon le nombre d'arguments rentrés
#define FILE_LOG1(level) \
    if (level > FILELOG_MAX_LEVEL) ;\
    else if (level > FILELog::ReportingLevel() || !Output2FILE::Stream()) ; \
    else FILELog().Get(level)

#define FILE_LOG2(level, obj) \
    if (level > FILELOG_MAX_LEVEL) ;\
    else if (level > FILELog::ReportingLevel() || !Output2FILE::Stream()) ; \
    else FILELog().Get(level, getClassName(obj))

#define FILE_LOG3(level, obj, funcName) \
    if (level > FILELOG_MAX_LEVEL) ;\
    else if (level > FILELog::ReportingLevel() || !Output2FILE::Stream()) ; \
    else FILELog().Get(level, getClassName(obj), funcName)


//! Choix de la macro appellée selon le nombre d'arguments
#define N_ARGS_IMPL3(_1, _2, _3, count, ...) \
   count
#define N_ARGS_IMPL2(args) \
    N_ARGS_IMPL3 args
#define N_ARGS_IMPL(args) \
   N_ARGS_IMPL3 args
#define N_ARGS(...) N_ARGS_IMPL((__VA_ARGS__, 3, 2, 1, 0))

#define CHOOSER3(count) FILE_LOG##count
#define CHOOSER2(count) CHOOSER3(count)
#define CHOOSER1(count) CHOOSER2(count)
#define CHOOSER(count)  CHOOSER1(count)

#define FILE_LOG_GLUE(x, y) x y
#define FILE_LOG(...) \
   FILE_LOG_GLUE(CHOOSER(N_ARGS(__VA_ARGS__)), \
               (__VA_ARGS__))

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)

#include <windows.h>

inline std::string NowTime()
{
    const int MAX_LEN = 200;
    char buffer[MAX_LEN];
    if (GetTimeFormatA(LOCALE_USER_DEFAULT, 0, 0,
            "HH':'mm':'ss", buffer, MAX_LEN) == 0)
        return "Error in NowTime()";

    char result[100] = {0};
    static DWORD first = GetTickCount();
    sprintf(result, "%s.%03ld", buffer, (long)(GetTickCount() - first) % 1000);
    return result;
}

#else

#include <sys/time.h>

inline std::string NowTime()
{
    char buffer[11];
    time_t t;
    time(&t);
    tm r ;
    r.tm_sec = 0;
        r.tm_min = 0;
        r.tm_hour = 0;
        r.tm_mon = 0;
        r.tm_year = 0;
        r.tm_wday = 0;
        r.tm_yday = 0;
        r.tm_isdst = 0;
    strftime(buffer, sizeof(buffer), "%X", localtime_r(&t, &r));
    struct timeval tv;
    gettimeofday(&tv, 0);
    char result[100] = {0};
    sprintf(result, "%s.%03ld", buffer, (long)tv.tv_usec / 1000);
    return result;
}


#endif //WIN32

#endif //__LOG_H__
