#include "OperatorSetColorOpengl.h"

#include<CData.h>
#include<DataOpenGl.h>
#include<DataNumber.h>

OperatorSetColorOpenGl::OperatorSetColorOpenGl(){
    this->path().push_back("Algorithm");
    this->path().push_back("Visualization");
    this->path().push_back("OpenGl");
    this->setKey("OperatorSetColorOpenGl");
    this->setName("color");
    this->setInformation("set the color of the opengl elements where red coresponds to (r=255,g=0,b=0), green (r=0,g=255,b=0) and blue (r=0,g=0,b=255) and any color a ratio of these three colors");
    this->structurePlug().addPlugIn(DataOpenGl::KEY,"g.OpenGl");
    this->structurePlug().addPlugIn(DataNumber::KEY,"r.num(by default 0)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"g.num(by default 0)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"b.num(by default 0)");
    this->structurePlug().addPlugOut(DataOpenGl::KEY,"f.OpenGl");
}
void OperatorSetColorOpenGl::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);

    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);

    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);

    if(this->plugIn()[3]->isConnected()==false)
        this->plugIn()[3]->setState(CPlug::OLD);
    else
        this->plugIn()[3]->setState(CPlug::EMPTY);


    this->plugOut()[0]->setState(CPlug::EMPTY);
}

void OperatorSetColorOpenGl::exec(){
    shared_ptr<Scene3d> f1 = dynamic_cast<DataOpenGl *>(this->plugIn()[0]->getData())->getData();

    double r=0;
    if(this->plugIn()[1]->isDataAvailable()==true)
        r = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();

    double g=0;
    if(this->plugIn()[2]->isDataAvailable()==true)
        g = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    double b=0;
    if(this->plugIn()[3]->isDataAvailable()==true)
        b = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();

    for(int i =0;i<(int)f1->vfigure.size();i++)
    {
        RGBUI8 c(r,g,b);
        f1->vfigure[i]->setRGB(c);
    }


    dynamic_cast<DataOpenGl *>(this->plugOut()[0]->getData())->setData(f1);
}
COperator * OperatorSetColorOpenGl::clone(){
    return new OperatorSetColorOpenGl();
}
