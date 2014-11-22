#include "OperatorGetPoint.h"

#include<CData.h>
#include<DataPoint.h>
#include<DataNumber.h>

OperatorGetPoint::OperatorGetPoint(){


    this->path().push_back("Data");
    this->path().push_back("Point");
    this->path().push_back("Tool");
    this->setKey("OperatorGetPoint");
    this->setName("getValue");
    this->setInformation("v=V(i)");

    this->structurePlug().addPlugIn(DataPoint::KEY,"V.v");
    this->structurePlug().addPlugIn(DataNumber::KEY,"i.num");
    this->structurePlug().addPlugOut(DataNumber::KEY,"v.num");
}

void OperatorGetPoint::exec(){
    VecF64  V = dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();
    int i = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    if(i>=0&&i<V.size()){
        double v = V.operator ()(i);
        dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(v);
    }else{
        string msg = "Out of range vector of size "+UtilityString::Any2String(V.size())+" and i is equal to "+UtilityString::Any2String(i);
        this->error(msg);
        return;
    }

}



COperator * OperatorGetPoint::clone(){
    return new OperatorGetPoint();
}
