#include "OperatorTypeImageGrid.h"

#include<DataImageGrid.h>
#include<DataString.h>
OperatorTypeMatN::OperatorTypeMatN()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Tool");
    this->setKey("PopulationOperatorTypeImageGrid");
    this->setName("type");
    this->setInformation("Get the type of the input image");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugOut(DataString::KEY,"type.str");
}



void OperatorTypeMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    BaseMatN * fc1= f1.get();
    string type;
    foo func;
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,type,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataString *>(this->plugOut()[0]->getData())->setValue(type);
}

COperator * OperatorTypeMatN::clone(){
    return new OperatorTypeMatN();
}
