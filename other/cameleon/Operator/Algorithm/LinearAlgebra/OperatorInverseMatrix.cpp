#include "OperatorInverseMatrix.h"
#include"algorithm/LinearAlgebra.h"

#include<CData.h>
#include<DataMatrix.h>
OperatorInverseMatrix::OperatorInverseMatrix(){

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"C.m");
    this->path().push_back("Algorithm");
    this->path().push_back("LinearAlgebra");
    this->setKey("OperatorInverseMatrix");
    this->setName("inverseGaussianElimination");
    this->setInformation("C=$A^\\{-1\\}$");
}

void OperatorInverseMatrix::exec(){
    shared_ptr<Mat2F64> m1 = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();

    try{
       Mat2F64* m = new  Mat2F64;
       *m = LinearAlgebra::inverseGaussianElimination(*m1);
       dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}



COperator * OperatorInverseMatrix::clone(){
    return new OperatorInverseMatrix();
}
