#include "OperatorPoissonDistribution.h"
#include<DataNumber.h>
OperatorPoissonDistribution::OperatorPoissonDistribution()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Catalog");
    this->path().push_back("Discrete");
    this->setKey("PopulationOperatorPoissonDistribution");
    this->setName("poisson");
    this->setInformation("f(n)=$\\exp(-\\lambda)\\frac\\{lambda^n\\}\\{n!\\}$");
    this->structurePlug().addPlugIn(DataNumber::KEY,"lambda.num");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"f.dist");
}

void OperatorPoissonDistribution::exec(){
    double lambda = dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    DistributionPoisson gen(lambda);
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(gen);

}

COperator * OperatorPoissonDistribution::clone(){
    return new OperatorPoissonDistribution();
}

