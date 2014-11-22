#include "OperatorPencilDistribution.h"

#include "data/distribution/DistributionFromDataStructure.h"
#include<DataMatrix.h>
OperatorPencilDistribution::OperatorPencilDistribution()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Catalog");
    this->path().push_back("Discrete");
    this->setKey("PopulationOperatorPencilDistribution");
    this->setName("pencil");
    this->setInformation("f is a pencil function defined by the input matrix first column is x and the second one f(x)\n");
    this->structurePlug().addPlugIn(DataMatrix::KEY,"matrix.m");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"f.dist");
}


void OperatorPencilDistribution::exec(){
    shared_ptr<Mat2F64> matrix = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    DistributionPencil gen;
    try{
    gen.fromMatrix(*(matrix.get()));
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(gen);
    }
    catch(pexception msg){
        this->error(msg.what());
    }

}

COperator * OperatorPencilDistribution::clone(){
    return new OperatorPencilDistribution();
}
