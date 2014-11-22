#include "OperatorMinImageGrid.h"

#include<DataImageGrid.h>
OperatorMinMatN::OperatorMinMatN()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Arithmetic");
    this->setKey("PopulationOperatorMinImageGrid");
    this->setName("minimum");
    this->setInformation("h=min(f1,f2)\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f1.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f2.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorMinMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> f2 = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    BaseMatN * fc2= f2.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,fc2,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        if(msg.what()[0]=='P')
            this->error("Pixel/voxel type of input image must be registered type");
        else
            this->error(msg.what());
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorMinMatN::clone(){
    return new OperatorMinMatN();
}
