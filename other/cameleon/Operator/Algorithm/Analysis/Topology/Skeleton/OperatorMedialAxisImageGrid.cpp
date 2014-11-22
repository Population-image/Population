#include "OperatorMedialAxisImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataString.h>
#include<DataMatrix.h>
OperatorMedialAxisMatN::OperatorMedialAxisMatN()
    :COperator()
{
        this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Topology");
    this->path().push_back("Skeleton");
    this->setKey("PopulationOperatorMedialAxisImageGrid");
    this->setName("medialAxis");
    this->setInformation("Thinning the binary image at constant topology\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"bin.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"norm.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"skeleton.pgm");
}
void OperatorMedialAxisMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);

    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);

    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorMedialAxisMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    int norm =1;
    if(this->plugIn()[1]->isDataAvailable()==true)
        norm = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    foo func;
    BaseMatN * skeleton;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid1Byte>::Switch(func,fc1,norm,skeleton,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(skeleton));
}
COperator * OperatorMedialAxisMatN::clone(){
    return new OperatorMedialAxisMatN();
}
