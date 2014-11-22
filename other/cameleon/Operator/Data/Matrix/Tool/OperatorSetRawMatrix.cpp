#include "OperatorSetRawMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorSetRawMatrix::OperatorSetRawMatrix(){

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugIn(DataPoint::KEY,"V.v");
    this->structurePlug().addPlugIn(DataNumber::KEY,"i.num");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"B.m");
    this->path().push_back("Data");
    this->path().push_back("Matrix");
        this->path().push_back("Tool");
    this->setKey("OperatorSetRawMatrix");
    this->setName("setRaw");
    this->setInformation("B(k,j)=A(k,j) for k neq i, V(j) otherwise");
}

void OperatorSetRawMatrix::exec(){
    shared_ptr<Mat2F64> m1 = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    VecF64  v = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
    int i = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    try{
        m1->setRow(i,v);
        dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m1));
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}

COperator * OperatorSetRawMatrix::clone(){
    return new OperatorSetRawMatrix();
}
