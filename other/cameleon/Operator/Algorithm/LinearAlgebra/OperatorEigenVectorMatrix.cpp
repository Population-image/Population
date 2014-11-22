#include "OperatorEigenVectorMatrix.h"

#include"algorithm/LinearAlgebra.h"
#include<CData.h>
#include<DataMatrix.h>
#include<DataNumber.h>
#include<DataPoint.h>
OperatorEigenVectorMatrix::OperatorEigenVectorMatrix(){

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugIn(DataPoint::KEY,"eigenvalue.num");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"C.m");
    this->path().push_back("Algorithm");
    this->path().push_back("LinearAlgebra");
    this->setKey("OperatorEigenVectorMatrix");
    this->setName("eigenVectorGaussianElimination");
    this->setInformation("Eigen Vectors contained in the columns of the output matrix by Gauss-Jordan's algorithm where the input vector is the eigen values");
}
void OperatorEigenVectorMatrix::exec(){
    shared_ptr<Mat2F64> m1 = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    VecF64  v = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
    try{
       *m1.get() = LinearAlgebra::eigenVecGaussianElimination(*m1.get(),v);
       dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(m1);
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}



COperator * OperatorEigenVectorMatrix::clone(){
    return new OperatorEigenVectorMatrix();
}
