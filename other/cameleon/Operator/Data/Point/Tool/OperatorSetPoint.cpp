#include "OperatorSetPoint.h"

#include<DataPoint.h>
#include<DataNumber.h>

OperatorSetPoint::OperatorSetPoint(){


    this->path().push_back("Data");
    this->path().push_back("Point");
        this->path().push_back("Tool");
    this->setKey("OperatorSetPoint");
    this->setName("setValue");
    this->setInformation("B(j)= a for i=j, A(j) otherwise");

    this->structurePlug().addPlugIn(DataPoint::KEY,"A.v");
    this->structurePlug().addPlugIn(DataNumber::KEY,"i.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"a.num");
    this->structurePlug().addPlugOut(DataPoint::KEY,"B.v");
}

void OperatorSetPoint::exec(){

    VecF64  V = dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();
    int i = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    double v = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    if(i>=0&&i<V.size()){
        V.operator ()(i)=v;
        dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(V);
    }else{
        this->error("Out of range the input index is"+UtilityString::Any2String(i)+" and the vector size is "+UtilityString::Any2String(V.size()));
    }
}



COperator * OperatorSetPoint::clone(){
    return new OperatorSetPoint();
}
