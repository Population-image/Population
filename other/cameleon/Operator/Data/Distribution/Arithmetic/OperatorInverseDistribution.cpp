#include "OperatorInverseDistribution.h"
#include"data/distribution/DistributionFromDataStructure.h"
#include<DataDistribution.h>
OperatorInverseDistribution::OperatorInverseDistribution()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Arithmetic");
    this->setKey("PopulationOperatorInverseDistribution");
    this->setName("inverse");
    this->setInformation("h=1/f\n");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"h.dist");
}

void OperatorInverseDistribution::exec(){
    Distribution f= dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();

    Distribution gen(DistributionExpression("1"));
    f= gen/f;
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(f);
}

COperator * OperatorInverseDistribution::clone(){
    return new OperatorInverseDistribution();
}
