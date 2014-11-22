#include "OperatorMarchingCubeColorImageGridOpenGl.h"

#include<DataImageGrid.h>
#include<DataOpenGl.h>

#include"algorithm/Visualization.h"
using namespace pop;
#include"algorithm/Visualization.h"
using namespace pop;
OperatorMarchingCubeColorImageGridOpenGl::OperatorMarchingCubeColorImageGridOpenGl()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Visualization");
    this->path().push_back("OpenGl");
    this->setKey("PopulationOperatorMarchingCubeColorImageGridOpenGl");
    this->setName("marchingCube");
    this->setInformation("Marching cube of the input 3D Color image ");
    this->structurePlug().addPlugIn(DataMatN::KEY,"Color.pgm");
    this->structurePlug().addPlugOut(DataOpenGl::KEY,"h.pgm");
}


void OperatorMarchingCubeColorImageGridOpenGl::exec(){
    shared_ptr<BaseMatN> bin = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    MatN<3,RGBUI8 >  * binc;
    bool deleteimage=false;
    if(dynamic_cast<MatN<3,RGBUI8 >  * >(bin.get())){
        binc = dynamic_cast<MatN<3,RGBUI8 >  * >(bin.get());
    }else if(MatN<3,unsigned char >  * grey =  dynamic_cast<MatN<3,unsigned char >  * >(bin.get())){
        binc = new MatN<3,RGBUI8 >(grey->getDomain());
        *binc =*grey;
        deleteimage =true;
    }else{
        int dim;
        string type;
        bin->getInformation(type, dim);
        this->error("The input image must be 3D with a voxel type coded in Color or 1Byte.\n Your input image is "+UtilityString::Any2String(dim)+" and the pixel/voxel type is "+type);
        return;
    }
    Scene3d * out = new Scene3d();
    Visualization::marchingCube(*out,*binc);
    dynamic_cast<DataOpenGl *>(this->plugOut()[0]->getData())->setData(shared_ptr<Scene3d>(out));
    if(deleteimage==true)
        delete binc;


}

COperator * OperatorMarchingCubeColorImageGridOpenGl::clone(){
    return new OperatorMarchingCubeColorImageGridOpenGl();
}
