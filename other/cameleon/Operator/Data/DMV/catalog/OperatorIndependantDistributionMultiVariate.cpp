#include "OperatorIndependantDistributionMultiVariate.h"

#include "data/distribution/DistributionMultiVariateFromDataStructure.h"
#include<DataString.h>
OperatorIndependantDistributionMultiVariate::OperatorIndependantDistributionMultiVariate()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("DistributionMultiVariate");
    this->path().push_back("Catalog");
    this->setKey("PopulationOperatorIndependantDistributionMultiVariate");
    this->setName("Independant");
    this->setInformation("Construct a distribution such that h(x,y)=f(x)*g(y) Idem for the probability distribution H(X={x,y})=F(X={x})G(Y={y})");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"g.dist");
    this->structurePlug().addPlugOut(DataDistributionMultiVariate::KEY,"h.dist");
}

void OperatorIndependantDistributionMultiVariate::exec(){
    DistributionMultiVariate f = dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[0]->getData())->getValue();
    DistributionMultiVariate g = dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[1]->getData())->getValue();
    DistributionMultiVariate gen(f,g);
    dynamic_cast<DataDistributionMultiVariate *>(this->plugOut()[0]->getData())->setValue(gen);
}

COperator * OperatorIndependantDistributionMultiVariate::clone(){
    return new OperatorIndependantDistributionMultiVariate();
}
