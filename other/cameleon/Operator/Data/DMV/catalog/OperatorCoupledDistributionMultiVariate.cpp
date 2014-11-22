#include "OperatorCoupledDistributionMultiVariate.h"

#include "data/distribution/DistributionMultiVariateFromDataStructure.h"
#include<DataDistribution.h>
#include<DataNumber.h>
OperatorCoupledDistributionMultiVariate::OperatorCoupledDistributionMultiVariate()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("DistributionMultiVariate");
    this->path().push_back("Catalog");
    this->setKey("PopulationOperatorCoupledDistributionMultiVariate");
    this->setName("coupled");
    this->setInformation("generate a random number H(X={x,...,z,})=F(X={x}) for x =...=z, 0 otherwise");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataNumber::KEY,"number.num");
    this->structurePlug().addPlugOut(DataDistributionMultiVariate::KEY,"h.dist");
}

void OperatorCoupledDistributionMultiVariate::exec(){
    Distribution f = dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();
    int n = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    DistributionMultiVariate gen(f,n);
    dynamic_cast<DataDistributionMultiVariate *>(this->plugOut()[0]->getData())->setValue(gen);
}

COperator * OperatorCoupledDistributionMultiVariate::clone(){
    return new OperatorCoupledDistributionMultiVariate();
}
