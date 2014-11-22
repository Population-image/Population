#include "OperatorThresholdImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorThresholdMatN::OperatorThresholdMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Point");
    this->setKey("PopulationOperatorThresholdImageGrid");
    this->setName("threshold");
    this->setInformation("h(x)=255 for v1<=f(x) <v2, 0 otherwise where 256 is the default value for v2 \n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"v1.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"v2.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}








void OperatorThresholdMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();

    double v1 = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    double v2;
    if(this->plugIn()[2]->isDataAvailable()==true){
        v2 = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    }
    else{
        v2 = 256;
    }

    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,v1,v2,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

void OperatorThresholdMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
COperator * OperatorThresholdMatN::clone(){
    return new OperatorThresholdMatN();
}
