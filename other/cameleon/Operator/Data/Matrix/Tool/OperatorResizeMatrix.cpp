#include "OperatorResizeMatrix.h"
#include<CData.h>
#include<DataMatrix.h>
#include<DataNumber.h>

OperatorResizeMatrix::OperatorResizeMatrix(){


    this->path().push_back("Data");
    this->path().push_back("Matrix");
        this->path().push_back("Tool");
    this->setKey("OperatorResizeMatrixMatrix");
    this->setName("resize");
    this->setInformation("B(i,j)=A(i,j) with nbr rows of B = sizei and  nbr cols of B = sizej");

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugIn(DataNumber::KEY,"sizei.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"sizej.num");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"B.num");
}

void OperatorResizeMatrix::exec(){
    shared_ptr<Mat2F64> m = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    int i = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    int j = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    if(i>=0 && j>=0){
        m->resize(i,j);
        this->plugOut()[0]->getData()->setMode(this->plugIn()[0]->getData()->getMode());
        dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(m);
    }else{
        this->error("Out of range");
    }
}



COperator * OperatorResizeMatrix::clone(){
    return new OperatorResizeMatrix();
}
