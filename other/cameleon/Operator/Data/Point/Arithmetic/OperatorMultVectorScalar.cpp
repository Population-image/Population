#include "OperatorMultVectorScalar.h"

#include<CData.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorMultVectorScalarVector::OperatorMultVectorScalarVector(){

    this->structurePlug().addPlugIn(DataPoint::KEY,"A.v");
    this->structurePlug().addPlugIn(DataNumber::KEY,"b.num");
    this->structurePlug().addPlugOut(DataPoint::KEY,"c.v");
    this->path().push_back("Data");
    this->path().push_back("Point");
    this->path().push_back("Arithmetic");

    this->setKey("OperatorMultVectorScalarPoint");
    this->setName("multiplicationScalar");
    this->setInformation("C(i)=A(i)*b");
}

void OperatorMultVectorScalarVector::exec(){
    VecF64  A = dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();
    double b = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();

    try{
       A*=b;
       dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(A);
    }
    catch(pexception msg){
        this->error(msg.what());
    }
}



COperator * OperatorMultVectorScalarVector::clone(){
    return new OperatorMultVectorScalarVector();
}
