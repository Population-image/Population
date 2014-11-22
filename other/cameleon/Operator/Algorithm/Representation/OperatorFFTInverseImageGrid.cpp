#include "OperatorFFTInverseImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorFFTInverseMatN::OperatorFFTInverseMatN()
    :COperator()
{
        this->path().push_back("Algorithm");
    this->path().push_back("Representation");
    this->setKey("PopulationOperatorFFTInverseImageGrid");
    this->setName("FFTInverse");
    this->setInformation("h=FFTInverse(f)\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}
void OperatorFFTInverseMatN::exec(){
    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    BaseMatN * h;
    foo func;
    BaseMatN * fc= f.get();
    try{Dynamic2Static<TListImgGridComplex>::Switch(func,fc,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorFFTInverseMatN::clone(){
    return new OperatorFFTInverseMatN();
}
