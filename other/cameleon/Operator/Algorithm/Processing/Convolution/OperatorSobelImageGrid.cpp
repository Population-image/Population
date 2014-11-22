#include "OperatorSobelImageGrid.h"

#include<DataImageGrid.h>
#include<DataNumber.h>
OperatorGradientNormSobelMatN::OperatorGradientNormSobelMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Convolution");
    this->setKey("PopulationOperatorGradientNormSobelImageGrid");
    this->setName("gradientMagnitudeSobel");
    this->setInformation("MatNGradNormSobel f.pgm h.pgm h(x)=|grad(g)| with grad=sobel\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f1.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}


void OperatorGradientNormSobelMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be registered type");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorGradientNormSobelMatN::clone(){
    return new OperatorGradientNormSobelMatN();
}
