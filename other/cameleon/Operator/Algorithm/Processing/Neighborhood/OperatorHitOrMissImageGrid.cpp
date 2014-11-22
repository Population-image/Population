#include "OperatorHitOrMissImageGrid.h"

#include<DataImageGrid.h>
OperatorHitOrMissMatN::OperatorHitOrMissMatN()
    :COperator()
{
        this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Neighborhood");
    this->setKey("PopulationOperatorHitOrMissImageGrid");
    this->setName("hitOrMiss");
    this->setInformation("h(x)=Erosion(f,C) Union Erosion(f$^c$,C) where f$^c$(x)=255 for f(x)=0, 0 otherwise ");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"C.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"D.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}


void OperatorHitOrMissMatN::exec(){
    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> C = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
    shared_ptr<BaseMatN> D = dynamic_cast<DataMatN *>(this->plugIn()[2]->getData())->getData();
    BaseMatN * h;
    foo func;

    BaseMatN * fc= f.get();
    BaseMatN * Cc= C.get();
    BaseMatN * Dc= D.get();
    try{Dynamic2Static<TListImgGridUnsigned>::Switch(func,fc,Cc,Dc,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type \n f, C and D must have the same dimension\n C and D must have the same pixel/voxel type");
        return;
    }

    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}
COperator * OperatorHitOrMissMatN::clone(){
    return new OperatorHitOrMissMatN();
}
