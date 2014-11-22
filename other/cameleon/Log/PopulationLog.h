#ifndef POPULATIONLOG_H
#define POPULATIONLOG_H

#include"data/utility/CollectorExecutionInformation.h"
using namespace pop;
using namespace std;
class PopulationLog : public CollectorExecutionInformationImplementation
{
private:
    string _libname;
public:
    PopulationLog(string libname="POP");

    virtual void startExecution(string function,CollectorExecutionMode mode);
    virtual  void info(string msg);
    virtual void endExecution(string function);
    virtual void progression(double ratio,string msg="");

};

#endif // POPULATIONLOG_H
