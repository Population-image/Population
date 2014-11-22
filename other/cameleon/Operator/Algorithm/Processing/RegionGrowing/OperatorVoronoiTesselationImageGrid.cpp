#include "OperatorVoronoiTesselationImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorVoronoiTesselationMatN::OperatorVoronoiTesselationMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("RegionGrowing");
    this->setKey("PopulationOperatorVoronoiTesselationImageGrid");
    this->setName("voronoiTesselation");
    this->setInformation("$h(x) = \\{ i : d(x,s_i)<=d(x,s_j),\\forall j\\}$ and $dist(x) = min_i d(x,s_i)$\n where $s_{j>0}=\\{x:seed(x)=j\\}$\n d(x,si) is the minimum distance between x and all point of si in the set (the euclidean norm doesn't work with a mask)");
    this->structurePlug().addPlugIn(DataMatN::KEY,"seed.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"mask.pgm(by default no mask)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"n.num(by default 1)");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"dist.pgm");
}


void OperatorVoronoiTesselationMatN::initState(){
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
    this->plugOut()[1]->setState(CPlug::EMPTY);

}

void OperatorVoronoiTesselationMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    int norm;
    if(this->plugIn()[2]->isDataAvailable()==true)
        norm = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    else
        norm = 1;
    BaseMatN * h;
    BaseMatN * dist;
    foo func;
    BaseMatN * fc1= f1.get();
    if(this->plugIn()[1]->isDataAvailable()==true){
        shared_ptr<BaseMatN>  f2 = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
        BaseMatN * fc2= f2.get();
        try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,fc1,fc2,norm,h,dist,Loki::Type2Type<MatN<2,int> >());}
        catch(pexception msg){
            if(msg.what()[0]=='P')
                this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
            else
                this->error(msg.what());
            return;
        }
    }
    else{
        try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,fc1,norm,h,dist,Loki::Type2Type<MatN<2,int> >());}
        catch(pexception msg){
            if(msg.what()[0]=='P')
                this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
            else
                this->error(msg.what());
            return;
        }
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
    dynamic_cast<DataMatN *>(this->plugOut()[1]->getData())->setData(shared_ptr<BaseMatN>(dist));
}

COperator * OperatorVoronoiTesselationMatN::clone(){
    return new OperatorVoronoiTesselationMatN();
}
