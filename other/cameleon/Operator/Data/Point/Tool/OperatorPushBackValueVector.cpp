#include "OperatorPushBackValueVector.h"

#include<CData.h>
#include<DataPoint.h>
#include<DataNumber.h>

OperatorPushBackPoint::OperatorPushBackPoint(){


    this->path().push_back("Data");
    this->path().push_back("Point");
        this->path().push_back("Tool");
    this->setKey("OperatorPushBackPoint");
    this->setName("pushBack");
    this->setInformation("B(j)=A(j) and  B(A.size())=value");

    this->structurePlug().addPlugIn(DataPoint::KEY,"A.m");
    this->structurePlug().addPlugIn(DataNumber::KEY,"value.num");
    this->structurePlug().addPlugOut(DataPoint::KEY,"B.num");
}

void OperatorPushBackPoint::exec(){
    VecF64  V = dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();
    double value = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    V.data().push_back(value);
    dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(V);
}



COperator * OperatorPushBackPoint::clone(){
    return new OperatorPushBackPoint();
}
