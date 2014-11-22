#include "OperatorOrthogonalMatrix.h"
#include"algorithm/LinearAlgebra.h"
#include<CData.h>
#include<DataMatrix.h>
OperatorOrthogonalMatrix::OperatorOrthogonalMatrix(){

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"C.m");
    this->path().push_back("Algorithm");
    this->path().push_back("LinearAlgebra");
    this->setKey("OperatorOrthogonalMatrix");
    this->setName("orthogonalGramSchmidt");
    this->setInformation("Gram–Schmidt process for the orthonormalising a set of vectors defined by the input matrix");
}

void OperatorOrthogonalMatrix::exec(){
    shared_ptr<Mat2F64> m1 = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    try{
       Mat2F64* m = new  Mat2F64;
       *m = LinearAlgebra::orthogonalGramSchmidt(*m1.get());
       dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m));
    }

    catch(pexception msg){
        this->error(msg.what());
    }
}



COperator * OperatorOrthogonalMatrix::clone(){
    return new OperatorOrthogonalMatrix();
}
