#include "OperatorRepartitionAreaLabelImageGrid.h"

#include<DataImageGrid.h>
#include<DataPoint.h>
OperatorRepartitionAreaLabelMatN::OperatorRepartitionAreaLabelMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Geometry");
    this->path().push_back("Label");
    this->setKey("PopulationOperatorRepartitionAreaLabelImageGrid");
    this->setName("area");
    this->setInformation("$V(i)=|\\{x:f(x)=i+1 \\}$, we count the area of each label");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugOut(DataPoint::KEY,"V.v");
}

void OperatorRepartitionAreaLabelMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    VecF64  v;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,v,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(v);
}

COperator * OperatorRepartitionAreaLabelMatN::clone(){
    return new OperatorRepartitionAreaLabelMatN();
}
