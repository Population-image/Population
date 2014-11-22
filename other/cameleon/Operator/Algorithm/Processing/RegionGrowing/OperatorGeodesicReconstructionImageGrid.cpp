

#include "OperatorGeodesicReconstructionImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorGeodesicReconstructionMatN::OperatorGeodesicReconstructionMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("RegionGrowing");
    this->setKey("PopulationOperatorGeodesicReconstructionImageGrid");
    this->setName("geodesicReconstruction");
    this->setInformation("$f_{i+1}=max(erosion(f_i),g)$ with $f_0 = f$ and $f_{\\infty}=h$ ");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"g.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"norm.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}
void OperatorGeodesicReconstructionMatN::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);

}


void OperatorGeodesicReconstructionMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN>  f2 = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
    int norm;
    if(this->plugIn()[2]->isDataAvailable()==true)
        norm = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    else
        norm = 1;
    BaseMatN * h;
    foo func;
    BaseMatN * fc1= f1.get();
    BaseMatN * fc2= f2.get();
    try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,fc1,fc2,norm,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        if(msg.what()[0]=='P')
            this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
        else
            this->error(msg.what());
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));

}

COperator * OperatorGeodesicReconstructionMatN::clone(){
    return new OperatorGeodesicReconstructionMatN();
}
