#include "OperatorIdentityMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataNumber.h>

#include"algorithm/LinearAlgebra.h"
OperatorIdentityMatrix::OperatorIdentityMatrix(){


    this->path().push_back("Algorithm");
    this->path().push_back("LinearAlgebra");
    this->setKey("OperatorIdentityMatrix");
    this->setName("identity");
    this->setInformation("A(i,j)=0 for i neq j, 1 otherwise with the a square size");
    this->structurePlug().addPlugIn(DataNumber::KEY,"size.num");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"A.m");

}

void OperatorIdentityMatrix::exec(){

    int size= dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();

    shared_ptr<Mat2F64> m(new Mat2F64(size,size));
    for(int i =0;i<m->sizeI();i++)
        m->operator ()(i,i)=1;
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(m);
}



COperator * OperatorIdentityMatrix::clone(){
    return new OperatorIdentityMatrix();
}
