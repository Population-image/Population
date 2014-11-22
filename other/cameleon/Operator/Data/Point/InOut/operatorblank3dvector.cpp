#include "operatorblank3dvector.h"

#include<CData.h>
#include<DataPoint.h>
#include<DataNumber.h>

OperatorBlank3DPoint::OperatorBlank3DPoint(){


    this->path().push_back("Data");
    this->path().push_back("Point");
    this->path().push_back("Tool");
    this->setKey("OperatorBlank3DPoint");
    this->setName("blank3D");
    this->setInformation("V(0)=vx, V(1)=vy and V(2)=vz");
    this->structurePlug().addPlugIn(DataNumber::KEY,"vx.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"vy.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"vz.num");
    this->structurePlug().addPlugOut(DataPoint::KEY,"V.v");
}

void OperatorBlank3DPoint::exec(){


    double v_x = dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    double v_y = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    double v_z = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    VecF64  m(3);
    m(0)=v_x;
    m(1)=v_y;
    m(2)=v_z;
    dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(m);
}



COperator * OperatorBlank3DPoint::clone(){
    return new OperatorBlank3DPoint();
}
