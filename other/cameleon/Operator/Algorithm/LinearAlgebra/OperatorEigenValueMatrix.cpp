#include "OperatorEigenValueMatrix.h"

#include"algorithm/LinearAlgebra.h"
#include<CData.h>
#include<DataMatrix.h>
#include<DataNumber.h>
#include<DataPoint.h>
OperatorEigenValueMatrix::OperatorEigenValueMatrix(){

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugIn(DataNumber::KEY,"error.num");
    this->structurePlug().addPlugOut(DataPoint::KEY,"C.m");
    this->path().push_back("Algorithm");
    this->path().push_back("LinearAlgebra");
    this->setKey("OperatorEigenValueMatrix");
    this->setName("eigenValueQR");
    this->setInformation("Eigen Values of the input matrix with the QR algorithm ");
}
void OperatorEigenValueMatrix::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorEigenValueMatrix::exec(){
    shared_ptr<Mat2F64> m1 = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    int error =0.01;
    if(this->plugIn()[1]->isDataAvailable()==true)
        error = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    try{
       VecF64  v = LinearAlgebra::eigenValueQR(*m1.get(),error);
       dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(v);
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}



COperator * OperatorEigenValueMatrix::clone(){
    return new OperatorEigenValueMatrix();
}
