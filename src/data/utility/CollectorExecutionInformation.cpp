//#include"data/utility/CollectorExecutionInformation.h"
//#include"data/utility/BasicUtility.h"

//namespace pop
//{

//CollectorExecutionInformationImplementation::CollectorExecutionInformationImplementation()

//{
//}
//CollectorExecutionInformationImplementation::~CollectorExecutionInformationImplementation(){

//}

//void CollectorExecutionInformationImplementation::startExecution(std::string function, CollectorExecutionType mode){
//    _mode=mode;
//    std::cout<<"start: "+function<<std::endl;
//}

//void CollectorExecutionInformationImplementation::endExecution(std::string function){
//    std::cout<<"end: "+function<<std::endl;
//}


//void CollectorExecutionInformationImplementation::progression(F64 ratio,std::string msg){
//    std::cout<<msg<<" Progression: "+BasicUtility::Any2String(ratio)<<std::endl;
//}
//void CollectorExecutionInformationImplementation::info(std::string msg){
//    std::cout<<msg<<std::endl;
//}

//Private::CollectorExecutionInformation::~CollectorExecutionInformation()
//{
//    delete _impl;
//}
//Private::CollectorExecutionInformation::CollectorExecutionInformation()
//    :_impl(new CollectorExecutionInformationImplementation()),_activate(false)
//{

//}
//void Private::CollectorExecutionInformation::setCollector(CollectorExecutionInformationImplementation* collector){
//    delete _impl;
//    _impl=collector;
//}

//void Private::CollectorExecutionInformation::startExecution(std::string function,CollectorExecutionType mode){
//    if(_activate==true)
//        _impl->startExecution(function,mode);
//}

//void Private::CollectorExecutionInformation::endExecution(std::string function){
//    if(_activate==true)
//        _impl->endExecution(function);
//}
//void Private::CollectorExecutionInformation::info(std::string msg){
//    if(_activate==true)
//        _impl->info(msg);
//}

//void Private::CollectorExecutionInformation::progression(F64 ratio,std::string msg){
//    if(_activate==true)
//        _impl->progression(ratio,msg);
//}
//void Private::CollectorExecutionInformation::setActivate(bool activate){

//    _activate = activate;
//}
//}
