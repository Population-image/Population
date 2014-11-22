#include "OperatorConvolutionImageGrid.h"

#include "OperatorConvolutionImageGrid.h"
#include<DataImageGrid.h>
OperatorConvolutionMatN::OperatorConvolutionMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Convolution");
    this->setKey("PopulationOperatorConvolutionImageGrid");
    this->setName("convolution");
    this->setInformation("$h(x)= \\int_O^{\\infty} f(x+x')k(-x')dx$");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataMatN::KEY,"k.pgm");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}
void OperatorConvolutionMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<BaseMatN> f2 = dynamic_cast<DataMatN *>(this->plugIn()[1]->getData())->getData();
    BaseMatN * h;
    foo func;
    BaseMatN * fc1= f1.get();
    BaseMatN * fc2= f2.get();
    try{Dynamic2Static<TListImgGrid>::Switch(func,fc1,fc2,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("f and struct_elt must have the same dimension");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorConvolutionMatN::clone(){
    return new OperatorConvolutionMatN();
}
