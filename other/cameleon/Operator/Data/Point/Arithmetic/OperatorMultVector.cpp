#include <OperatorMultVector.h>
#include<DataNumber.h>
#include<DataPoint.h>

OperatorMultVector::OperatorMultVector(){
    this->structurePlug().addPlugIn(DataPoint::KEY,"A.v");
    this->structurePlug().addPlugIn(DataPoint::KEY,"B.v");
    this->structurePlug().addPlugOut(DataNumber::KEY,"c.num");
    this->path().push_back("Data");
    this->path().push_back("Point");
    this->path().push_back("Arithmetic");
    this->setKey("OperatorInnerProductPoint");
    this->setName("productInner");
    this->setInformation("c=sum$_i$ A(i) * B(i) ");
}

void OperatorMultVector::exec(){
    VecF64  a = dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();
    VecF64  b = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
    try{
        double c=productInner(a,b);
        dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(c);
    }catch(pexception msg){
        this->error(msg.what());
    }
}

COperator * OperatorMultVector::clone(){
    return new OperatorMultVector;
}
