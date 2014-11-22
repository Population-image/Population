#include "OperatorGetMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataNumber.h>
OperatorGetMatrix::OperatorGetMatrix(){


    this->path().push_back("Data");
    this->path().push_back("Matrix");
        this->path().push_back("Tool");
    this->setKey("OperatorGetMatrix");
    this->setName("getValue");
    this->setInformation("a=A(i,j)");

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugIn(DataNumber::KEY,"i.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"j.num");
    this->structurePlug().addPlugOut(DataNumber::KEY,"a.num");
}

void OperatorGetMatrix::exec(){
    shared_ptr<Mat2F64> m = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    int i = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    int j = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    if(m->isValid(i,j)){
        double a = m->operator ()(i,j);
        dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(a);
    }
    else{
        this->error("Out of range of i or j");
    }
}



COperator * OperatorGetMatrix::clone(){
    return new OperatorGetMatrix();
}
