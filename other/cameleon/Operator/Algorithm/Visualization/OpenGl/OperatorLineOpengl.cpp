#include "OperatorLineOpengl.h"

OperatorLineOpenGl::OperatorLineOpenGl()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Visualization");
    this->path().push_back("OpenGl");
    this->setKey("PopulationOperatorLineOpenGl");
    this->setName("lineCube");
    this->setInformation("Boundary lines of the input 3D binary image ");
    this->structurePlug().addPlugIn(DataMatN::KEY,"binary.pgm");
    this->structurePlug().addPlugOut(DataOpenGl::KEY,"h.pgm");
}

void OperatorLineOpenGl::exec(){
    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();

    foo func;
    Scene3d * out = new Scene3d();
    BaseMatN * fc= f.get();
    typedef  FilterKeepTlistTlist<TListImgGrid,0,Loki::Int2Type<3> >::Result ListFilter;
    typedef FilterRemoveTlistTlist<ListFilter, 1, Complex<pop::F64> >::Result ListFilter2;

    try{Dynamic2Static<ListFilter2>::Switch(func,fc,out,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataOpenGl *>(this->plugOut()[0]->getData())->setData(shared_ptr<Scene3d>(out));

}

COperator * OperatorLineOpenGl::clone(){
    return new OperatorLineOpenGl();
}
