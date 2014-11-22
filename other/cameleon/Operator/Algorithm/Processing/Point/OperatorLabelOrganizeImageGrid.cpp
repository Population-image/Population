#include "OperatorLabelOrganizeImageGrid.h"

#include<DataImageGrid.h>
OperatorLabelOrganizeMatN::OperatorLabelOrganizeMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Point");
    this->setKey("PopulationOperatorLabelOrganizeImageGrid");
    this->setName("greylevelRemoveEmptyValue");
    this->setInformation("Contract the grey-level dynamic such that any grey-level without any pixel of this level in the  input image is removed\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorLabelOrganizeMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorLabelOrganizeMatN::clone(){
    return new OperatorLabelOrganizeMatN();
}
