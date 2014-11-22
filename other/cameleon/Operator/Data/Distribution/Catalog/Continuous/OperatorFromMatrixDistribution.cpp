#include "OperatorFromMatrixDistribution.h"
#include "data/distribution/DistributionFromDataStructure.h"
#include<DataMatrix.h>
OperatorFromMatrixDistribution::OperatorFromMatrixDistribution()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Catalog");
    this->path().push_back("Continuous");
    this->setKey("PopulationOperatorFromMatrixDistribution");
    this->setName("regularStep");
    this->setInformation("f is a step function defined by the input matrix first column is x and the second one f(x)\n");
    this->structurePlug().addPlugIn(DataMatrix::KEY,"matrix.m");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"f.dist");
}

void OperatorFromMatrixDistribution::exec(){
    shared_ptr<Mat2F64> matrix = dynamic_cast<DataMatrix*>(this->plugIn()[0]->getData())->getData();
    DistributionRegularStep gen;
    try{
    gen.fromMatrix(*(matrix.get()));
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(gen);
    }
    catch(pexception msg){
        this->error(msg.what());
    }

}

COperator * OperatorFromMatrixDistribution::clone(){
    return new OperatorFromMatrixDistribution();
}
