#include "OperatorSetColMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorSetColMatrix::OperatorSetColMatrix(){

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugIn(DataPoint::KEY,"V.v");
    this->structurePlug().addPlugIn(DataNumber::KEY,"j.num");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"B.m");
    this->path().push_back("Data");
    this->path().push_back("Matrix");
        this->path().push_back("Tool");
    this->setKey("OperatorSetColMatrix");
    this->setName("setCol");
    this->setInformation("B(i,k)=A(i,k) for k neq k, V(i) otherwise");
}

void OperatorSetColMatrix::exec(){
    shared_ptr<Mat2F64> m1 = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    VecF64  v = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
    int j = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    try{
        m1->setCol(j,v);
        dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m1));
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}

COperator * OperatorSetColMatrix::clone(){
    return new OperatorSetColMatrix();
}
