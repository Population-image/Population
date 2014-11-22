#include "OperatorSetPlaneImageGrid.h"



#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorsetPlaneMatN::OperatorsetPlaneMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Affect");
    this->setKey("PopulationOperatorsetPlaneImageGrid");
    this->setName("setPlane");
    this->setInformation("the ouput image is equal to the input image except the plane for the given coordinate and the given index");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"plane.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"index.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"coordinate.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}
void OperatorsetPlaneMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugIn()[2]->setState(CPlug::EMPTY);
    if(this->plugIn()[3]->isConnected()==false)
        this->plugIn()[3]->setState(CPlug::OLD);
    else
        this->plugIn()[3]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorsetPlaneMatN::exec(){
   shared_ptr<BaseMatN> f     = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> plane = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();

    BaseMatN * h;
    foo func;

    BaseMatN * fc= f.get();
    BaseMatN * planec= plane.get();
    int index = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    int coordinate=0;
    if(this->plugIn()[3]->isDataAvailable()==true){
        coordinate = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();
    }
    typedef  FilterKeepTlistTlist<TListImgGrid,0,Loki::Int2Type<3> >::Result ListFilter;
    try{Dynamic2Static<ListFilter>::Switch(func,fc,planec,index,coordinate,h,Loki::Type2Type<MatN<2,int> >());}
    catch(string msg){
        this->error("Pixel/voxel type of input image topo must be registered type");
        return;
    }


    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorsetPlaneMatN::clone(){
    return new OperatorsetPlaneMatN();
}
