#include "OperatorSizeMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataNumber.h>

OperatorSizeMatrix::OperatorSizeMatrix(){


    this->path().push_back("Data");
    this->path().push_back("Matrix");
        this->path().push_back("Tool");
    this->setKey("OperatorSizeMatrix");
    this->setName("domain");
    this->setInformation("sizei is the number of rows of A and sizej the number of cols");

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugOut(DataNumber::KEY,"sizei.num");
    this->structurePlug().addPlugOut(DataNumber::KEY,"sizej.num");
}

void OperatorSizeMatrix::exec(){
    shared_ptr<Mat2F64> m = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();

    dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(m->sizeI());
    dynamic_cast<DataNumber *>(this->plugOut()[1]->getData())->setValue(m->sizeJ());
}



COperator * OperatorSizeMatrix::clone(){
    return new OperatorSizeMatrix();
}
