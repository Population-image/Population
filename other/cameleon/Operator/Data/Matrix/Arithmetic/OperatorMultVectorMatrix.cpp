#include "OperatorMultVectorMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataPoint.h>
OperatorMultMatrixVectorMatrix::OperatorMultMatrixVectorMatrix(){

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugIn(DataPoint::KEY,"b.v");
    this->structurePlug().addPlugOut(DataPoint::KEY,"c.v");
    this->path().push_back("Data");
    this->path().push_back("Matrix");
    this->path().push_back("Arithmetic");
    this->setKey("OperatorMultMatrixVectorMatrix");
    this->setName("multiplicationPoint");
    this->setInformation("c=A*b");
}

void OperatorMultMatrixVectorMatrix::exec(){
    shared_ptr<Mat2F64> A = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    VecF64  b = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();

    try{
       VecF64   c ;
       c = *(A.get())* b;
       dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(c);
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}



COperator * OperatorMultMatrixVectorMatrix::clone(){
    return new OperatorMultMatrixVectorMatrix();
}
