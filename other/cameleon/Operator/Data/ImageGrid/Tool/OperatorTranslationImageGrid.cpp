#include "OperatorTranslationImageGrid.h"
#include<DataImageGrid.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorTranslationMatN::OperatorTranslationMatN()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Tool");
    this->setKey("PopulationOperatorTranslationImageGrid");
    this->setName("translation");
    this->setInformation("h(x)= f(x+t)\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataPoint::KEY,"t.v");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorTranslationMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    VecF64  xmin =    dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();

    BaseMatN * h;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,xmin,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}


COperator * OperatorTranslationMatN::clone(){
    return new OperatorTranslationMatN();
}

