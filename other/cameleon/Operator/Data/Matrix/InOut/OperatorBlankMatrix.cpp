#include "OperatorBlankMatrix.h"

#include<CData.h>
#include<DataMatrix.h>
#include<DataNumber.h>

OperatorBlankMatrix::OperatorBlankMatrix(){


    this->path().push_back("Data");
    this->path().push_back("Matrix");
    this->path().push_back("Tool");
    this->setKey("OperatorBlankMatrix");
    this->setName("blank");
    this->setInformation("A(i,j)=value with dimension sizei and sizej");
    this->structurePlug().addPlugIn(DataNumber::KEY,"sizei.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"sizej.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"value.num(by default O)");

    this->structurePlug().addPlugOut(DataMatrix::KEY,"A.m");

}
void OperatorBlankMatrix::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);

    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorBlankMatrix::exec(){

    int sizei= dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    int sizej= dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();

    double value=0;
    if(this->plugIn()[2]->isDataAvailable()==true)
        value = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    shared_ptr<Mat2F64> m(new Mat2F64(sizei,sizej,value));
    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(m);
}



COperator * OperatorBlankMatrix::clone(){
    return new OperatorBlankMatrix();
}
