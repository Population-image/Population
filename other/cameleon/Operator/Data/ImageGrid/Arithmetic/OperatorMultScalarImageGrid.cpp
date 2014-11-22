#include "OperatorMultScalarImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorMultScalarMatN::OperatorMultScalarMatN()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Arithmetic");
    this->setKey("PopulationOperatorMultScalarImageGrid");
    this->setName("multiplicationScalar");
    this->setInformation("h(x)=v*f(x)\n");
    this->structurePlug().addPlugIn(DataNumber::KEY,"v.num");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}
void OperatorMultScalarMatN::exec(){
    double v = dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    shared_ptr<BaseMatN> f2 = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
    BaseMatN * h;
    foo func;


    BaseMatN * fc2= f2.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc2,v,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorMultScalarMatN::clone(){
    return new OperatorMultScalarMatN();
}
