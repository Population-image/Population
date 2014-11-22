#include "OperatorfofxImageGrid.h"

#include<DataImageGrid.h>
#include<DataDistribution.h>
OperatorfofxMatN::OperatorfofxMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Point");
    this->setKey("PopulationOperatorfofxImageGrid");
    this->setName("fofx");
    this->setInformation("h(x)=P(f(x))\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"P.dist");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorfofxMatN::exec(){
    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<Distribution> P = dynamic_cast<DataDistribution *>(this->plugIn()[1]->getData())->getData();
    BaseMatN * h;
    foo func;

    BaseMatN * fc= f.get();
    Distribution * Pc= P.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc,Pc,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be scalar");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorfofxMatN::clone(){
    return new OperatorfofxMatN();
}
