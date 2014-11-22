#include "OperatorRandomFieldImageGrid.h"

#include<DataImageGrid.h>
#include<DataDistribution.h>
OperatorRandomFieldMatN::OperatorRandomFieldMatN()
    :COperator()
{

    this->path().push_back("Algorithm");
    this->path().push_back("Processing");
    this->path().push_back("Generator");
    this->setKey("PopulationOperatorRandomField");
    this->setName("randomField");
    this->setInformation("h(x)=X~P where X is random variable following the law P and domain(h)=domain(f)\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"f.pgm");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"P.dist");
    this->structurePlug().addPlugOut(DataMatN::KEY,"h.pgm");
}


void OperatorRandomFieldMatN::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();
    Distribution d1 = dynamic_cast<DataDistribution *>(this->plugIn()[1]->getData())->getValue();
    BaseMatN * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,d1,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be scalar");
        return;
    }
    dynamic_cast<DataMatN *>(this->plugOut()[0]->getData())->setData(shared_ptr<BaseMatN>(h));
}

COperator * OperatorRandomFieldMatN::clone(){
    return new OperatorRandomFieldMatN();
}
