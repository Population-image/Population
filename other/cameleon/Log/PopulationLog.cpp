#include "PopulationLog.h"

#include<CLogger.h>
#include<CMachine.h>
#include"data/utility/UtilitySTL.h"
PopulationLog::PopulationLog(string libname)
    :_libname(libname)
{

    CLogger::getInstance()->registerComponent(_libname.c_str());
    CLogger::getInstance()->setComponentLevel(_libname.c_str(),CLogger::INFO);
}

void PopulationLog::startExecution(string function,CollectorExecutionMode mode){


        CLogger::getInstance()->log(_libname.c_str(),CLogger::INFO,function.c_str());


}
void PopulationLog::info(string msg){


        CLogger::getInstance()->log(_libname.c_str(),CLogger::INFO,msg.c_str());

}

void PopulationLog::endExecution(string function){


        CLogger::getInstance()->log(_libname.c_str(),CLogger::INFO,function.c_str());


}

void PopulationLog::progression(double ratio,string msg){

        string str =msg+" Progression: "+UtilityString::Any2String(ratio);

        CLogger::getInstance()->log(_libname.c_str(),CLogger::INFO,str.c_str());

        CMachineSingleton::getInstance()->getProcessor()->progressionExecution(ratio);
}
