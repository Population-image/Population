#include "OperatorGetColMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorGetColMatrix::OperatorGetColMatrix(){

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugIn(DataNumber::KEY,"j.num");
    this->structurePlug().addPlugOut(DataPoint::KEY,"V.v");
    this->path().push_back("Data");
    this->path().push_back("Matrix");
        this->path().push_back("Tool");
    this->setKey("OperatorGetColMatrix");
    this->setName("getCol");
    this->setInformation("V(i)= A(i,j)");
}

void OperatorGetColMatrix::exec(){
    shared_ptr<Mat2F64> m1 = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    int j = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    try{
        VecF64  v = m1->getCol(j);
        dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(v);
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}

COperator * OperatorGetColMatrix::clone(){
    return new OperatorGetColMatrix();
}
