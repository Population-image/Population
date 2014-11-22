#include "OperatorTransparencyOpengl.h"
#include<DataNumber.h>
OperatorTransparencyOpenGl::OperatorTransparencyOpenGl()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Visualization");
    this->path().push_back("OpenGl");
    this->setKey("PopulationOperatorTransparencyOpenGl");
    this->setName("transparency");
    this->setInformation("activate the transparency mode and set the transparency objects contained in the input scene between [0,1] of ");
    this->structurePlug().addPlugIn(DataOpenGl::KEY,"scene1.opengl");
    this->structurePlug().addPlugIn(DataNumber::KEY,"t.num");
    this->structurePlug().addPlugOut(DataOpenGl::KEY,"sceneout.opengl");
}
void OperatorTransparencyOpenGl::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);

    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);

    this->plugOut()[0]->setState(CPlug::EMPTY);
}

void OperatorTransparencyOpenGl::exec(){
    shared_ptr<Scene3d> f1 = dynamic_cast<DataOpenGl *>(this->plugIn()[0]->getData())->getData();
    double transparency  =0.5;

    if(this->plugIn()[1]->isDataAvailable()==true)
        transparency = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    if(transparency<0)
    {
        this->error("Transparency value must belong to the range [0,1]");
        return;
    }
    if(transparency>1)
    {
        this->error("Transparency value must belong to the range [0,1]");
        return;
    }
    for(int i =0;i<(int)f1->vfigure.size();i++)
    {
        f1->vfigure[i]->setTransparent(255*transparency);
    }
    f1->transparency_mode=0;
    dynamic_cast<DataOpenGl *>(this->plugOut()[0]->getData())->setData(f1);


}

COperator * OperatorTransparencyOpenGl::clone(){
    return new OperatorTransparencyOpenGl();
}
