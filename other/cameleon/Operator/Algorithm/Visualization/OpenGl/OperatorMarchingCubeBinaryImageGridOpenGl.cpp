#include "OperatorMarchingCubeBinaryImageGridOpenGl.h"

#include<DataImageGrid.h>
#include<DataOpenGl.h>

#include"algorithm/Visualization.h"
using namespace pop;
OperatorMarchingCubeBinaryImageGridOpenGl::OperatorMarchingCubeBinaryImageGridOpenGl()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Visualization");
    this->path().push_back("OpenGl");
    this->setKey("PopulationOperatorMarchingCubeBinaryImageGridOpenGl");
    this->setName("marchingCube");
    this->setInformation("Marching cube of the input 3D binary image ");
    this->structurePlug().addPlugIn(DataMatN::KEY,"binary.pgm");
    this->structurePlug().addPlugOut(DataOpenGl::KEY,"h.pgm");
}

void OperatorMarchingCubeBinaryImageGridOpenGl::exec(){
    shared_ptr<BaseMatN> bin = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();

    if(MatN<3,unsigned char>  * binc = dynamic_cast<MatN<3,unsigned char>  * >(bin.get())){
        Scene3d * out = new Scene3d();
        Visualization::marchingCube(*out,*binc);
        dynamic_cast<DataOpenGl *>(this->plugOut()[0]->getData())->setData(shared_ptr<Scene3d>(out));
    }else
    {
        this->error("The input imahe must be 3D with a voxel type coded in 1 byte");
    }


}

COperator * OperatorMarchingCubeBinaryImageGridOpenGl::clone(){
    return new OperatorMarchingCubeBinaryImageGridOpenGl();
}
