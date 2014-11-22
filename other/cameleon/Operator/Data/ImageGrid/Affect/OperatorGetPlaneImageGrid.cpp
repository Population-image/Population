#include "OperatorGetPlaneImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorGetPlaneMatN::OperatorGetPlaneMatN()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Affect");
    this->setKey("PopulationOperatorGetPlaneImageGrid");
    this->setName("getPlane");
    this->setInformation("Extract the plane of the given index for the given coordinate of the input 3d image\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"index.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"coordinate.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorGetPlaneMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}

void OperatorGetPlaneMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    int index = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    int coordinate=-1;
    if(this->plugIn()[2]->isDataAvailable()==true){
        coordinate = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    }
    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();

    typedef  FilterRemoveTlistTlist<TListImgGrid,0,Loki::Int2Type<1> >::Result ListFilter;
    try{Dynamic2Static<ListFilter>::Switch(func,fc1,index,coordinate,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorGetPlaneMatN::clone(){
    return new OperatorGetPlaneMatN();
}
