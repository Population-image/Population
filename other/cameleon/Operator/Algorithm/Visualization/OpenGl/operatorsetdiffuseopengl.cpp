#include "operatorsetdiffuseopengl.h"

#include<DataNumber.h>
OperatorDiffuseOpenGl::OperatorDiffuseOpenGl()
    :COperator()
{


    this->path().push_back("Algorithm");
    this->path().push_back("Visualization");
    this->path().push_back("OpenGl");
    this->setKey("PopulationOperatorDiffuseOpenGl");
    this->setName("diffuse");
    this->setInformation("Set the ambiance contained in the input scene between red=[0,1], g=[0,1] and b=[0,1]");
    this->structurePlug().addPlugIn(DataOpenGl::KEY,"scene1.opengl");
    this->structurePlug().addPlugIn(DataNumber::KEY,"r.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"g.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"b.num");
    this->structurePlug().addPlugOut(DataOpenGl::KEY,"sceneout.opengl");
}


void OperatorDiffuseOpenGl::exec(){
    shared_ptr<Scene3d> f1 = dynamic_cast<DataOpenGl *>(this->plugIn()[0]->getData())->getData();
    double r  = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    if(r<0)
    {
        this->error("Diffuse value must belong to the range [0,1]");
        return;
    }
    if(r>1)
    {
        this->error("Diffuse value must belong to the range [0,1]");
        return;
    }
    double g  = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    if(g<0)
    {
        this->error("Diffuse value must belong to the range [0,1]");
        return;
    }
    if(g>1)
    {
        this->error("Diffuse value must belong to the range [0,1]");
        return;
    }

    double b  = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();
    if(b<0)
    {
        this->error("Diffuse value must belong to the range [0,1]");
        return;
    }
    if(b>1)
    {
        this->error("Diffuse value must belong to the range [0,1]");
        return;
    }


    f1->diffuse=RGB<double>(r,g,b);
    dynamic_cast<DataOpenGl *>(this->plugOut()[0]->getData())->setData(f1);


}

COperator * OperatorDiffuseOpenGl::clone(){
    return new OperatorDiffuseOpenGl();
}
