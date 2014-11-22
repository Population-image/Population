#include "OperatorSavePoint.h"

#include<CData.h>
#include<DataPoint.h>
#include<DataString.h>

#include<DataBoolean.h>

OperatorSavePoint::OperatorSavePoint(){
    this->structurePlug().addPlugIn(DataPoint::KEY,"A.v");
    this->structurePlug().addPlugIn(DataString::KEY,"file.str");
        this->structurePlug().addPlugOut(DataBoolean::KEY,"out.bool");
    this->path().push_back("Data");
    this->path().push_back("Point");
    this->path().push_back("InOut");
    this->setKey("OperatorSavePoint");
    this->setName("save");
    this->setInformation("Save point to the given file, out= false for bad writing, true otherwise");
}

void OperatorSavePoint::exec(){
    VecF64  v = dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();
    string str  = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    v.save(str);
    dynamic_cast<DataBoolean *>(this->plugOut()[0]->getData())->setValue(true);
}



COperator * OperatorSavePoint::clone(){
    return new OperatorSavePoint();
}
