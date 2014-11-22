#include "OperatorCorrelationDirectionByFFTImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
#include<DataMatrix.h>

OperatorCorrelationDirectionByFFTMatN::OperatorCorrelationDirectionByFFTMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Analysis");
    this->path().push_back("Geometry");
    this->path().push_back("Statistic");
    this->setKey("PopulationOperatorCorrelationDirectionByFFTImageGrid");
    this->setName("correlationDirectionByFFT");
    this->setInformation("C is the 2-point correlation function where C=FFT$^\\{-1\\}$(FFT(f)FFT(f)$^*$) with double foxel type");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"C.pgm");
}

void OperatorCorrelationDirectionByFFTMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    BaseMatN * fout;
    foo func;
    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,fout,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(fout));
}
COperator * OperatorCorrelationDirectionByFFTMatN::clone(){
    return new OperatorCorrelationDirectionByFFTMatN();
}
