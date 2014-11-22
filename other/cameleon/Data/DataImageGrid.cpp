#include "DataImageGrid.h"

const bool _registerMatNsingleton = GPFactoryRegister<_TListImgGrid>::Register(*SingletonFactoryMatN::getInstance(),Loki::Type2Type<MatN<2,F64> >());
DataMatN::DataMatN()
    :CDataByFile<BaseMatN>()
{
    this->_key = DataMatN::KEY;
    this->setExtension(".pgm");
}
string DataMatN::KEY ="DATAImageGrid";
DataMatN * DataMatN::clone(){
    return new DataMatN();
}

shared_ptr<BaseMatN> DataMatN::getDataByFile(){
    shared_ptr<BaseMatN> t(BaseMatN::create(this->getFile()) );
    return t;
}

//ImageD_UC * DataMatN::getImageD_UC()throw(pexception){
//    Image * t = this->getDataPointer();
//    if(ImageD_UC * ptr= dynamic_cast<ImageD_UC *>(t))
//        return ptr;
//    else{
//        string type;
//        int dim;
//        t->getInformation(type,dim);
//        throw(std::string("The image is not a 2D with unsigned 1Byte as pixe type but "+UtilityString::Any2String(dim)+"D and "+type +"as pixel type" ));
//    }
//}

//Image3D_UC * DataMatN::getImage3D_UC()throw(pexception){
//    Image * t = this->getDataPointer();
//    if(Image3D_UC * ptr= dynamic_cast<Image3D_UC *>(t))
//        return ptr;
//    else{
//        string type;
//        int dim;
//        t->getInformation(type,dim);
//        throw(std::string("The image is not a 3D with unsigned 1Byte as pixe type but "+UtilityString::Any2String(dim)+"D and "+type +"as pixel type" ));
//    }

//}
//ImageD_Color * DataMatN::getImageD_Color()throw(pexception){
//    Image * t = this->getDataPointer();
//    if(ImageD_Color * ptr= dynamic_cast<ImageD_Color *>(t))
//        return ptr;
//    else{
//        string type;
//        int dim;
//        t->getInformation(type,dim);
//        throw(std::string("The image is not a 2D with Color as pixel type but "+UtilityString::Any2String(dim)+"D and "+type +"as pixel type" ));
//    }
//}
//Image3D_Color * DataMatN::getImage3D_Color()throw(pexception){
//    Image * t = this->getDataPointer();
//    if(Image3D_Color * ptr= dynamic_cast<Image3D_Color *>(t))
//        return ptr;
//    else{
//        string type;
//        int dim;
//        t->getInformation(type,dim);
//        throw(std::string("The image is not a 3D with Color as pixel type but "+UtilityString::Any2String(dim)+"D and "+type +"as pixel type" ));
//    }
//}

void DataMatN::setDataByFile(shared_ptr<BaseMatN> type){

    type->save(this->getFile().c_str());
}
void DataMatN::setDataByCopy(shared_ptr<BaseMatN> type){

    this->_data = shared_ptr<BaseMatN> (new BaseMatN(*(type.get())));
}
