#include "OperatorMultCoordinateVector.h"

#include<CData.h>
#include<DataPoint.h>
#include<DataNumber.h>

OperatorMultCoordinatePoint::OperatorMultCoordinatePoint(){


    this->path().push_back("Data");
    this->path().push_back("Point");
        this->path().push_back("Tool");
    this->setKey("OperatorMultCoordinatePoint");
    this->setName("multiplicationCoordinate");
    this->setInformation("c = mult$_i$ V(i)");

    this->structurePlug().addPlugIn(DataPoint::KEY,"V.v");
    this->structurePlug().addPlugOut(DataNumber::KEY,"c.num");
}

void OperatorMultCoordinatePoint::exec(){
    VecF64  V = dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();
    dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(V.multCoordinate());

}



COperator * OperatorMultCoordinatePoint::clone(){
    return new OperatorMultCoordinatePoint();
}
