#include "OperatorAdamsBischofImageGrid.h"



#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataBoolean.h>
OperatorAdamsBischofMatN::OperatorAdamsBischofMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("RegionGrowing");
    this->setKey("PopulationOperatorAdamsBischofImageGrid");
    this->setName("regionGrowingAdamsBisch");
    this->setInformation("AdamsBischof algorithm for mode=0 $delta(x,i)= |topo(x)- <topo(x)>_{x in s_i}|$ and   for mode=1 $delta(x,i)= |topo(x)- <topo(x)>_{x in s_i}|/variance()$ ");
    this->structurePlug().addPlugIn(DataMatN::KEY,"seed.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"topo.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"n.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"mode.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"region.pgm");
}
void OperatorAdamsBischofMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
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


void OperatorAdamsBischofMatN::exec(){
    shared_ptr<BaseMatN> seed = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> topo = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
    int norm=0;
    if(this->plugIn()[2]->isDataAvailable()==true)
        norm = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    int mode=0;
    if(this->plugIn()[3]->isDataAvailable()==true)
        mode = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();


    BaseMatN *region;
    BaseMatN * seedc= seed.get();
    BaseMatN * topoc= topo.get();
    foo func;
    try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,seedc,topoc,norm,mode,region,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        if(msg.what()[0]=='P')
            this->error("Pixel/voxel type of input image must be scalar type");
        else
            this->error(msg.what());
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(region));
}

COperator * OperatorAdamsBischofMatN::clone(){
    return new OperatorAdamsBischofMatN();
}
