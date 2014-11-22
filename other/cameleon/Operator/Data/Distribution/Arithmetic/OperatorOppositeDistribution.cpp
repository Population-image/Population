#include "OperatorOppositeDistribution.h"
#include<DataDistribution.h>
#include"data/distribution/DistributionFromDataStructure.h"
OperatorOppositeDistribution::OperatorOppositeDistribution()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Arithmetic");
    this->setKey("PopulationOperatorOppositeDistribution");
    this->setName("oppoiste");
    this->setInformation("h=-f\n");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"h.dist");
}

void OperatorOppositeDistribution::exec(){
    Distribution f= dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();

    Distribution gen(DistributionExpression("0"));
    f= gen -f;
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(f);
}

COperator * OperatorOppositeDistribution::clone(){
    return new OperatorOppositeDistribution();
}
