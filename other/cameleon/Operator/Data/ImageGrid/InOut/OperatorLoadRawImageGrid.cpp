#include "OperatorLoadRawImageGrid.h"

#include<DataImageGrid.h>
#include<DataPoint.h>
#include<DataNumber.h>
#include<DataString.h>

OperatorLoadRawMatN::OperatorLoadRawMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("InOut");
    this->setKey("PopulationOperatorLoadRawImageGrid");
    this->setName("loadRaw");
    this->setInformation("LoadRaw image by file where size is the image size and \n type=P5 for 2d image in unsigned 1Byte,\n type=PA for 2d image in signed 2Bytes,\n type=PB for 2d image in signed 4Bytes\n type=PC for 2d image in float 8Bytes,\n type=P6 or 2d image in Color in 3*1Byte,\n type=PL for 3d image in unsigned 1Byte,\n type=PN for 2d image in signed 2Bytes,\n type=PP for 2d image in signed 4Bytes\n type=PR for 2d image in float 8Bytes,\n type=PT or 2d image in Color in 3*1Byte");
    this->structurePlug().addPlugIn(DataString::KEY,"file.str");
    this->structurePlug().addPlugIn(DataPoint::KEY,"size.v");
    this->structurePlug().addPlugIn(DataString::KEY,"type.str");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorLoadRawMatN::exec(){

    string file = dynamic_cast<DataString *>(this->plugIn()[0]->getData())->getValue();
    VecF64  domain = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
    string type = dynamic_cast<DataString *>(this->plugIn()[2]->getData())->getValue();
    try{
        BaseMatN * base;
        base =SingletonFactoryMatN::getInstance()->createObject(type);

        foo func;
        Dynamic2Static<TListImgGrid>::Switch(func,base,file,domain ,Loki::Type2Type<MatN<2,int> >());
        dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(base));
    }
    catch(pexception msg){
        this->error(msg.what());
    }

}

COperator * OperatorLoadRawMatN::clone(){
    return new OperatorLoadRawMatN();
}
