#include "OperatorRandomNonUniformPointGrainList.h"


OperatorRandomNonUniformPointGermGrain::OperatorRandomNonUniformPointGermGrain()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("Germ");
    this->setKey("OperatorRandomNonUniformPointGrainList");
    this->setName("poissonPointProcessNonUniform");
    this->setInformation("phi=$\\{x_0,...,x_\\{n-1\\}\\}$ is collection of random points following the lambda field\n");
    this->structurePlug().addPlugIn(DataMatN::KEY,"lambda.pgm");
    this->structurePlug().addPlugOut(DataGermGrain::KEY,"phi.grainlist");
}


void OperatorRandomNonUniformPointGermGrain::exec(){
    shared_ptr<BaseMatN> f1 = dynamic_cast<DataMatN *>(this->plugIn()[0]->getData())->getData();


    GermGrainMother * h;
    foo func;

    BaseMatN * fc1= f1.get();
    try{Dynamic2Static<TListImgGridScalar>::Switch(func,fc1,h,Loki::Type2Type<MatN<2,int> >());}
    catch(pexception msg){
        this->error("Pixel/voxel type of input image must be scalar");
        return;
    }
    dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(shared_ptr<GermGrainMother>(h));
}

COperator * OperatorRandomNonUniformPointGermGrain::clone(){
    return new OperatorRandomNonUniformPointGermGrain;
}
