#include "OperatorLabelAddImageGrid.h"

#include<DataImageGrid.h>
OperatorLabelAddMatN::OperatorLabelAddMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Seed");
    this->setKey("PopulationOperatorLabelAddImageGrid");
    this->setName("labelMerge");
    this->setInformation("h(x)=f(x) for g(x)=0, maxvalue(f)+g(x) otherswise");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"g.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorLabelAddMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> f2 = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    BaseMatN * fc2= f2.get();
    try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,fc1,fc2,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        if(msg.what()[0]=='P')
            this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
        else
            this->error(msg.what());
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorLabelAddMatN::clone(){
    return new OperatorLabelAddMatN();
}
