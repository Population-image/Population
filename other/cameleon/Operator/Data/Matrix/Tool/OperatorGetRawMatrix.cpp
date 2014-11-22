#include "OperatorGetRawMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorGetRawMatrix::OperatorGetRawMatrix(){

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugIn(DataNumber::KEY,"i.num");
    this->structurePlug().addPlugOut(DataPoint::KEY,"V.v");
    this->path().push_back("Data");
    this->path().push_back("Matrix");
        this->path().push_back("Tool");
    this->setKey("OperatorGetRawMatrix");
    this->setName("getRaw");
    this->setInformation("V(j)= A(i,j)");
}

void OperatorGetRawMatrix::exec(){
    shared_ptr<Mat2F64> m1 = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    int j = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    try{
        VecF64  v = m1->getRow(j);
        dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(v);
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}

COperator * OperatorGetRawMatrix::clone(){
    return new OperatorGetRawMatrix();
}
