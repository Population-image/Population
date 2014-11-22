#include "OperatorMultMatrixScalar.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataNumber.h>
OperatorMultMatrixScalarMatrix::OperatorMultMatrixScalarMatrix(){

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugIn(DataNumber::KEY,"b.num");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"C.m");
    this->path().push_back("Data");
    this->path().push_back("Matrix");
    this->path().push_back("Arithmetic");
    this->setKey("OperatorMultMatrixScalarMatrix");
    this->setName("multiplicationScalar");
    this->setInformation("C=A*b");
}

void OperatorMultMatrixScalarMatrix::exec(){
    shared_ptr<Mat2F64> A = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    double b = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();

    try{
       *(A.get())*=b;
       dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(A);
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}



COperator * OperatorMultMatrixScalarMatrix::clone(){
    return new OperatorMultMatrixScalarMatrix();
}
