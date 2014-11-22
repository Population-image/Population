#include "OperatorHeteregeneousRandomFieldImageGrid.h"
#include<DataDistribution.h>
#include<DataImageGrid.h>
OperatorHeteregeneousRandomFieldMatN::OperatorHeteregeneousRandomFieldMatN()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Point");
    this->setKey("PopulationOperatorHeteregeneousRandomFieldImageGrid");
    this->setName("randomFieldHeteregeneous");
    this->setInformation("h(x)=X~P(f(x)) where X is a random variable following the probability distribution P(f(x))\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"P.dist");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}

void OperatorHeteregeneousRandomFieldMatN::exec(){
    shared_ptr<BaseMatN> f = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    Distribution P = dynamic_cast<DataDistribution *>(this->plugIn()[1]->getData())->getValue();
    BaseMatN * h;
    foo func;

    BaseMatN * fc= f.get();

    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc,P,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be scalar");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorHeteregeneousRandomFieldMatN::clone(){
    return new OperatorHeteregeneousRandomFieldMatN();
}
