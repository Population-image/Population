#include "OperatorSizePoint.h"

#include<CData.h>
#include<DataPoint.h>
#include<DataNumber.h>

OperatorSizePoint::OperatorSizePoint(){


    this->path().push_back("Data");
    this->path().push_back("Point");
    this->path().push_back("Tool");
    this->setKey("OperatorSizePoint");
    this->setName("domain");
    this->setInformation("size is the number of elements of V");

    this->structurePlug().addPlugIn(DataPoint::KEY,"V.v");
    this->structurePlug().addPlugOut(DataNumber::KEY,"size.num");
}

void OperatorSizePoint::exec(){
    VecF64  m = dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();

    dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(m.size());
}



COperator * OperatorSizePoint::clone(){
    return new OperatorSizePoint();
}
