#include "OperatorLoadPoint.h"

#include<CData.h>
#include<DataPoint.h>
#include<DataString.h>

OperatorLoadPoint::OperatorLoadPoint(){

    this->structurePlug().addPlugIn(DataString::KEY,"file.str");
    this->structurePlug().addPlugOut(DataPoint::KEY,"A.v");
    this->path().push_back("Data");
    this->path().push_back("Point");
    this->path().push_back("InOut");
    this->setKey("OperatorLoadPoint");
    this->setName("load");
    this->setInformation("Load Point from file");
}

void OperatorLoadPoint::exec(){
    string str  = dynamic_cast<DataString *>(this->plugIn()[0]->getData())->getValue();
    VecF64  v;
    v.load(str);
    dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(v);
}



COperator * OperatorLoadPoint::clone(){
    return new OperatorLoadPoint();
}
