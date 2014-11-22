#include "OperatorSetMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataNumber.h>

OperatorSetMatrix::OperatorSetMatrix(){


    this->path().push_back("Data");
    this->path().push_back("Matrix");
        this->path().push_back("Tool");
    this->setKey("OperatorSetMatrix");
    this->setName("setValue");
    this->setInformation("B(k,p)= a for i=k and j=p, A(k,p) otherwise");

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugIn(DataNumber::KEY,"i.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"j.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"a.num");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"B.m");
}

void OperatorSetMatrix::exec(){
    shared_ptr<Mat2F64> m = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    int i = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    int j = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    double value = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();

    m->operator ()(i,j)=value;

    this->plugOut()[0]->getData()->setMode(this->plugIn()[0]->getData()->getMode());
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
}



COperator * OperatorSetMatrix::clone(){
    return new OperatorSetMatrix();
}
