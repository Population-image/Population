#include "OperatorDiracDistribution.h"

#include<DataNumber.h>
OperatorDiracDistribution::OperatorDiracDistribution()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Catalog");
    this->path().push_back("Continuous");
    this->setKey("PopulationOperatorDiracDistribution");
    this->setName("dirac");
    this->setInformation("As function f(x)=1/step for v-step/2<=x<v+step/2, 0 otherwise \n as probability distribution P(X=x)=1 for x = v, 0 otherwise\n");
    this->structurePlug().addPlugIn(DataNumber::KEY,"v.num");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"P.dist");
}

void OperatorDiracDistribution::exec(){
    double v = dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    DistributionDirac gen(v);

    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(gen);

}

COperator * OperatorDiracDistribution::clone(){
    return new OperatorDiracDistribution();
}
