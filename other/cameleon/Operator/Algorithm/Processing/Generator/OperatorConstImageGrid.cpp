#include "OperatorConstImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorConstMatN::OperatorConstMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Generator");
    this->setKey("PopulationOperatorConst");
    this->setName("fill");
    this->setInformation("h(x)=c where domain(h) = domain(f)\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"c.num");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorConstMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    double value = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,value,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorConstMatN::clone(){
    return new OperatorConstMatN();
}
