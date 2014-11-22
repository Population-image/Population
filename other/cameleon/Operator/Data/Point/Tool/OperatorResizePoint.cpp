#include "OperatorResizePoint.h"

#include<CData.h>
#include<DataPoint.h>
#include<DataNumber.h>

OperatorResizePoint::OperatorResizePoint(){


    this->path().push_back("Data");
    this->path().push_back("Point");
        this->path().push_back("Tool");
    this->setKey("OperatorResizePoint");
    this->setName("resize");
    this->setInformation("B(j)=0 for j>=size(A), A(j) otherwise");

    this->structurePlug().addPlugIn(DataPoint::KEY,"V.m");
    this->structurePlug().addPlugIn(DataNumber::KEY,"size.num");
    this->structurePlug().addPlugOut(DataPoint::KEY,"V'.num");
}

void OperatorResizePoint::exec(){
    VecF64  V = dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();
    int sizei = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    V.resize(sizei);
    dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(V);
}



COperator * OperatorResizePoint::clone(){
    return new OperatorResizePoint();
}
