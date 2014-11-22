#include "OperatorFromDistributionDistributionMultiVariate.h"

#include "data/distribution/DistributionMultiVariateFromDataStructure.h"
#include<DataDistribution.h>
#include<DataNumber.h>
OperatorFromDistributionDistributionMultiVariate::OperatorFromDistributionDistributionMultiVariate()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("DistributionMultiVariate");
    this->path().push_back("Catalog");
    this->setKey("PopulationOperatorFromDistributionDistributionMultiVariate");
    this->setName("fromDistribution");
    this->setInformation("generate a random number H(X={x})=F(X={x})");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugOut(DataDistributionMultiVariate::KEY,"h.dist");
}

void OperatorFromDistributionDistributionMultiVariate::exec(){
    Distribution f = dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();
    DistributionMultiVariate gen(f);
    dynamic_cast<DataDistributionMultiVariate *>(this->plugOut()[0]->getData())->setValue(gen);
}

COperator * OperatorFromDistributionDistributionMultiVariate::clone(){
    return new OperatorFromDistributionDistributionMultiVariate();
}
