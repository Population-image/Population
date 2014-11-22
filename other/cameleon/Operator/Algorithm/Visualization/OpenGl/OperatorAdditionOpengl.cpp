#include "OperatorAdditionOpengl.h"




OperatorAdditionOpenGl::OperatorAdditionOpenGl()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Visualization");
    this->path().push_back("OpenGl");
    this->setKey("PopulationOperatorAdditionOpenGl");
    this->setName("Addition");
    this->setInformation("Addition of the two opengl scene");
    this->structurePlug().addPlugIn(DataOpenGl::KEY,"scene1.opengl");
    this->structurePlug().addPlugIn(DataOpenGl::KEY,"scene2.opengl");
    this->structurePlug().addPlugOut(DataOpenGl::KEY,"sceneout.opengl");
}


void OperatorAdditionOpenGl::exec(){
    shared_ptr<Scene3d> f1 = dynamic_cast<DataOpenGl *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<Scene3d> f2 = dynamic_cast<DataOpenGl *>(this->plugIn()[1]->getData())->getData();


    shared_ptr<Scene3d> f3 (new Scene3d);
    for(int i =0;i<(int)f1->vfigure.size();i++)
    {
        f3->vfigure.push_back(f1->vfigure[i]->clone());
    }


    for(int i =0;i<(int)f2->vfigure.size();i++)
    {
        f3->vfigure.push_back(f2->vfigure[i]->clone());
    }
    if(f1->transparency_mode==1||f2->transparency_mode==1){
        f3->transparency_mode=1;
    }
    dynamic_cast<DataOpenGl *>(this->plugOut()[0]->getData())->setData(f3);


}

COperator * OperatorAdditionOpenGl::clone(){
    return new OperatorAdditionOpenGl();
}
