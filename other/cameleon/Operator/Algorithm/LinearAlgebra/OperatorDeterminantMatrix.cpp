#include "OperatorDeterminantMatrix.h"

#include<DataMatrix.h>
#include<DataNumber.h>

OperatorDeterminantMatrix::OperatorDeterminantMatrix(){

    this->path().push_back("Algorithm");
    this->path().push_back("LinearAlgebra");
    this->setKey("OperatorDeterminantMatrix");
    this->setName("determinant");
    this->setInformation("Determinant of the matrix");

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugOut(DataNumber::KEY,"t.num");
}

void OperatorDeterminantMatrix::exec(){
    shared_ptr<Mat2F64> m = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    try{
    dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(m->determinant());
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}



COperator * OperatorDeterminantMatrix::clone(){
    return new OperatorDeterminantMatrix();
}
