#include "OperatorBlankVector.h"

#include<CData.h>
#include<DataPoint.h>
#include<DataNumber.h>

OperatorBlankPoint::OperatorBlankPoint(){


    this->path().push_back("Data");
    this->path().push_back("Point");
    this->path().push_back("Tool");
    this->setKey("OperatorBlankPoint");
    this->setName("blank");
    this->setInformation("V(i)=value with the given size");
    this->structurePlug().addPlugIn(DataNumber::KEY,"size.num");
        this->structurePlug().addPlugIn(DataNumber::KEY,"value.num(by default O)");
    this->structurePlug().addPlugOut(DataPoint::KEY,"V.v");
}
void OperatorBlankPoint::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);

    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);

    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorBlankPoint::exec(){


    int size = dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    double value=0;
    if(this->plugIn()[1]->isDataAvailable()==true)
        value = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();

    VecF64  m(size,value);
    dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(m);
}



COperator * OperatorBlankPoint::clone(){
    return new OperatorBlankPoint();
}
