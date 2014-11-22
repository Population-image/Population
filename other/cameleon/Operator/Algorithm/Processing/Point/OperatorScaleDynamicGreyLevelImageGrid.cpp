#include "OperatorScaleDynamicGreyLevelImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorScaleDynamicGreyLevelMatN::OperatorScaleDynamicGreyLevelMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Point");
    this->setKey("PopulationOperatorScaleDynamicGreyLevelImageGrid");
    this->setName("greylevelRange");
    this->setInformation("h(x)=(f(x)-min(f))*(max-min)/(max(f)-min(f))+min");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"min.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"max.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}
void OperatorScaleDynamicGreyLevelMatN::initState(){
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
void OperatorScaleDynamicGreyLevelMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();

    double minv=numeric_limits<double>::max();
    if(this->plugIn()[1]->isDataAvailable()==true)
        minv = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    double maxv=numeric_limits<double>::max();
    if(this->plugIn()[2]->isDataAvailable()==true)
        maxv = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();


    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,minv,maxv,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be scalar type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorScaleDynamicGreyLevelMatN::clone(){
    return new OperatorScaleDynamicGreyLevelMatN();
}

