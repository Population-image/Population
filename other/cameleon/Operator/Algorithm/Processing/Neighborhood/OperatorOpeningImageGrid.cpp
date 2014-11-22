#include "OperatorOpeningImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorOpeningMatN::OperatorOpeningMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Neighborhood");
    this->setKey("PopulationOperatorOpeningImageGrid");
    this->setName("opening");
    this->setInformation("h(x)=h(x)=Erosion(Dilation(f)) with a ball with radius r and norm n (n=2 by default)\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f1.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"r.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"n.num(by default 2)");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorOpeningMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorOpeningMatN::exec(){

    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    double r = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    double n;
    if(this->plugIn()[2]->isDataAvailable()==true){
        n = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    }
    else{
        n = 2;
    }
    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,r,n,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorOpeningMatN::clone(){
    return new OperatorOpeningMatN();
}
