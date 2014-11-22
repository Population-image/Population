#include "OperatorExponentielDistribution.h"
#include<DataNumber.h>
OperatorExponentielDistribution::OperatorExponentielDistribution()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Catalog");
    this->path().push_back("Continuous");
    this->setKey("PopulationOperatorExponentielDistribution");
    this->setName("exponential");
    this->setInformation("f(x)=$ \\lambda e^\\{-\\lambda x\\}$ if x>0;0 otherwise\n");
    this->structurePlug().addPlugIn(DataNumber::KEY,"lambda.num");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"f.dist");
}

void OperatorExponentielDistribution::exec(){
    double lambda = dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    DistributionExponential gen (lambda);
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(gen);

}

COperator * OperatorExponentielDistribution::clone(){
    return new OperatorExponentielDistribution();
}
