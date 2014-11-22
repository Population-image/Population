#include "DataOpenGl.h"

DataOpenGl::DataOpenGl()
    :CDataByFile<Scene3d>()
{
    _data = shared_ptr<Scene3d>(new Scene3d);
    this->_key = DataOpenGl::KEY;
    this->setExtension(".opengl");
    this->setMode(CData::BYADDRESS);
}
string DataOpenGl::KEY ="DATAOpenGl";
DataOpenGl * DataOpenGl::clone(){
    return new DataOpenGl();
}


shared_ptr<Scene3d> DataOpenGl::getDataByFile(){
    Scene3d * f  = new Scene3d();
    std::ifstream  in(this->getFile().c_str());
    f->load(in);
    return     shared_ptr<Scene3d> (f);
}



void DataOpenGl::setDataByFile(shared_ptr<Scene3d> type){

    std::ofstream  out(this->getFile().c_str());
    type->save(out);

}
void DataOpenGl::setDataByCopy(shared_ptr<Scene3d> type){
    Scene3d * phiin = new Scene3d;
    * phiin= *type.get();
    this->_data = shared_ptr<Scene3d>(phiin);
}
