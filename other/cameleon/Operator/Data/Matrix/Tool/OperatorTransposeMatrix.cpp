#include "OperatorTransposeMatrix.h"
#include<CData.h>
#include<DataMatrix.h>
OperatorTransposeMatrixx::OperatorTransposeMatrixx(){

    this->structurePlug().addPlugIn(DataMatrix::KEY,"A.m");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"C.m");
    this->path().push_back("Data");
    this->path().push_back("Matrix");
        this->path().push_back("Tool");
    this->setKey("OperatorTransposeMatrix");
    this->setName("transpose");
    this->setInformation("C=$A^t$");
}

void OperatorTransposeMatrixx::exec(){
    shared_ptr<Mat2F64> m1 = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();

    try{
       m1->transpose();
       dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(m1));
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}



COperator * OperatorTransposeMatrixx::clone(){
    return new OperatorTransposeMatrixx();
}
