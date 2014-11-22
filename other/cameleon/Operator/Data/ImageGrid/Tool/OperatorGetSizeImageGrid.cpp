#include "OperatorGetSizeImageGrid.h"
#include<DataImageGrid.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorGetSizeMatN::OperatorGetSizeMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Tool");
    this->setKey("PopulationOperatorGetSizeImageGrid");
    this->setName("domain");
    this->setInformation("d = domain(h)\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"h.pgm");
    this->structurePlug().addPlugOut(DataPoint::KEY,"d.v");

}

void OperatorGetSizeMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    VecF64  x ;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,x,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(x);
}

COperator * OperatorGetSizeMatN::clone(){
    return new OperatorGetSizeMatN();
}

