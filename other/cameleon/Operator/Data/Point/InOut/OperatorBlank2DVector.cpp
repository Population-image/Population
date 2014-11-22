#include "OperatorBlank2DVector.h"

#include<CData.h>
#include<DataPoint.h>
#include<DataNumber.h>

OperatorBlank2DPoint::OperatorBlank2DPoint(){


    this->path().push_back("Data");
    this->path().push_back("Point");
        this->path().push_back("Tool");
    this->setKey("OperatorBlank2DPoint");
    this->setName("blank2D");
    this->setInformation("V(0)=vx and V(1)=vy");
    this->structurePlug().addPlugIn(DataNumber::KEY,"vx.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"vy.num");
    this->structurePlug().addPlugOut(DataPoint::KEY,"V.v");
}

void OperatorBlank2DPoint::exec(){


    double v_x = dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    double v_y = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    VecF64  m(2);
    m(0)=v_x;
    m(1)=v_y;
    dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(m);
}



COperator * OperatorBlank2DPoint::clone(){
    return new OperatorBlank2DPoint();
}
