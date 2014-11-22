#include "OperatorGetPlaneOpengl.h"
#include<DataNumber.h>
OperatorPlaneOpenGl::OperatorPlaneOpenGl()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Visualization");
    this->path().push_back("OpenGl");
    this->setKey("PopulationOperatorPlaneOpenGl");
    this->setName("plane");
    this->setInformation("get a plane of the core sample in the given coordinate and in the given index");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"index.pgm(by default 0)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"coordinate.pgm(by default 2=z-direction)");
    this->structurePlug().addPlugOut(DataOpenGl::KEY,"h.pgm");
}
void OperatorPlaneOpenGl::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);

    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);

    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);

    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorPlaneOpenGl::exec(){
    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    int index = 0;
    if(this->plugIn()[1]->isDataAvailable()==true){
        index = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    }
    int coordinate=2;
    if(this->plugIn()[2]->isDataAvailable()==true){
        coordinate = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    }
    foo func;
    Scene3d * out = new Scene3d();
    BaseMatN * fc= f.get();
    typedef  FilterKeepTlistTlist<TListImgGrid,0,Loki::Int2Type<3> >::Result ListFilter;
    typedef FilterRemoveTlistTlist<ListFilter, 1, Complex<pop::F64> >::Result ListFilter2;


    try{Dynamic2Static<ListFilter2>::Switch(func,fc,index,coordinate,out,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataOpenGl *>(this->plugOut()[0]->getData())->setData(shared_ptr<Scene3d>(out));

}

COperator * OperatorPlaneOpenGl::clone(){
    return new OperatorPlaneOpenGl();
}
