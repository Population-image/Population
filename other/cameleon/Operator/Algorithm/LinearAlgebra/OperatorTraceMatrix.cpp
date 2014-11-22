#include "OperatorTraceMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataNumber.h>

OperatorTraceMatrix::OperatorTraceMatrix(){


    this->path().push_back("Algorithm");
    this->path().push_back("LinearAlgebra");
    this->setKey("OperatorTraceMatrix");
    this->setName("trace");
    this->setInformation("Trace of the matrix");

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugOut(DataNumber::KEY,"t.num");
}

void OperatorTraceMatrix::exec(){
    shared_ptr<Mat2F64> m = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    try{
    dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(m->trace());
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}



COperator * OperatorTraceMatrix::clone(){
    return new OperatorTraceMatrix();
}
