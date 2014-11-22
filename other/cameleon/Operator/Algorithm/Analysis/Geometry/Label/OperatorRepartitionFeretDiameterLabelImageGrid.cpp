#include "OperatorRepartitionFeretDiameterLabelImageGrid.h"

#include<DataImageGrid.h>
#include<DataMatrix.h>
#include<DataNumber.h>
#include<DataPoint.h>
OperatorRepartitionFeretDiameterLabelMatN::OperatorRepartitionFeretDiameterLabelMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Geometry");
    this->path().push_back("Label");
    this->setKey("PopulationOperatorRepartitionFeretDiameterLabelImageGrid");
    this->setName("feretDiameter");
    this->setInformation("FeretDiameters of each label (for norm=0 , D= 1/n*sum$_i$ diameter(i) and otherwise D= (mult$_i$ diameter(i))$^{1/n}$ ");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"norm.num");
    this->structurePlug().addPlugOut(DataPoint::KEY,"A.m");
}
void OperatorRepartitionFeretDiameterLabelMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    int norm = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    VecF64  v;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,norm,v,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(v);
}

COperator * OperatorRepartitionFeretDiameterLabelMatN::clone(){
    return new OperatorRepartitionFeretDiameterLabelMatN();
}
