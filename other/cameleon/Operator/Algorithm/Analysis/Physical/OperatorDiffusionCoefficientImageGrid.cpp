#include "OperatorDiffusionCoefficientImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataMatrix.h>
OperatorDiffusionSelfCoefficientMatN::OperatorDiffusionSelfCoefficientMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Physical");
    this->setKey("PopulationOperatorDiffusionSelfCoefficientImageGrid");
    this->setName("randomWalk");
    this->setInformation("self diffusion coefficient of normalized self D(t)/D$_0$ \n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f1.pgm");
    this->structurePlug().addPlugIn(DataNumber::KEY,"nbrwalker.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"timemax.num");
    this->structurePlug().addPlugOut(DataMatrix::KEY,"diffusion.m");
}
void OperatorDiffusionSelfCoefficientMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    int nbrwalker = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    int timemax=dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    Mat2F64* h;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid1Byte>::Switch(func,fc1,nbrwalker,timemax,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be 1Byte");
        return;
    }

    dynamic_cast<DataMatrix*>(this->plugOut()[0]->getData())->setData(shared_ptr<Mat2F64>(h));
}

COperator * OperatorDiffusionSelfCoefficientMatN::clone(){
    return new OperatorDiffusionSelfCoefficientMatN();
}
