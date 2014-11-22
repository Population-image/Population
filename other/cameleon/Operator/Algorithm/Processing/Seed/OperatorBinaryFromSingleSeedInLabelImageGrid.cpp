#include "OperatorBinaryFromSingleSeedInLabelImageGrid.h"
#include<DataImageGrid.h>
OperatorBinaryFromSingleSeedInLabelMatN::OperatorBinaryFromSingleSeedInLabelMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Seed");
    this->setKey("PopulationOperatorBinaryFromSingleSeedInLabelImageGrid");
    this->setName("labelFromSingleSeed");
    this->setInformation("h(x)=255  for label(x) = i, 0 otherwise  with  i the label value such that the bin image is included in it");
    this->structurePlug().addPlugIn(DataMatN::KEY,"bin.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"label.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorBinaryFromSingleSeedInLabelMatN::exec(){
    shared_ptr<BaseMatN> bin = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> label = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
    BaseMatN * h;
    foo func;

    BaseMatN * binc= bin.get();
    BaseMatN * labelc= label.get();
    try{Dynamic2Static<TListImgGrid1Byte>::Switch(func,labelc,binc,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        if(msg.what()[0]=='P')
            this->error("Pixel/voxel type of input image must be unsigned type used operator Convert1Byte");
        else
            this->error(msg.what());
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorBinaryFromSingleSeedInLabelMatN::clone(){
    return new OperatorBinaryFromSingleSeedInLabelMatN();
}
