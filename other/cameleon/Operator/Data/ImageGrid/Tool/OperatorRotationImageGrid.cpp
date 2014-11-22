#include "OperatorRotationImageGrid.h"

#include<DataImageGrid.h>
#include<DataMatrix.h>
#include<DataNumber.h>
OperatorRotationMatN::OperatorRotationMatN()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("ImageGrid");
    this->path().push_back("Tool");
//    this->path().push_back("Deterministic");
    this->setKey("PopulationOperatorRotationImageGrid");
    this->setName("rotation");
    this->setInformation("h(x)= f(R(x-center)+center) with center the center of the image domain\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataMatrix::KEY,"R.m");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorRotationMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<Mat2F64> m =    dynamic_cast<DataMatrix*>(this->plugIn()[1]->getData())->getData();

    BaseMatN * h;
    foo func;
    BaseMatN * fc1= f1.get();
    Mat2F64* mc = m.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1, mc, h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}


COperator * OperatorRotationMatN::clone(){
    return new OperatorRotationMatN();
}
