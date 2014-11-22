#include "OperatorMinVector.h"

#include<DataNumber.h>
#include<DataPoint.h>

OperatorMinVector::OperatorMinVector(){
    this->structurePlug().addPlugIn(DataPoint::KEY,"a.v");
    this->structurePlug().addPlugIn(DataPoint::KEY,"b.v");
    this->structurePlug().addPlugOut(DataPoint::KEY,"c.v");
    this->path().push_back("Data");
        this->path().push_back("Point");
    this->path().push_back("Arithmetic");
    this->setKey("OperatorMinPoint");
    this->setName("minimum");
    this->setInformation("c(i)=Min(a(i),b(i))");
}

void OperatorMinVector::exec(){
    VecF64  a = dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();
    VecF64  b = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
    try{
        VecF64  c=std::min(a,b);
        dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(c);
    }catch(pexception msg){
        this->error(msg.what());
    }


}

COperator * OperatorMinVector::clone(){
    return new OperatorMinVector;
}
