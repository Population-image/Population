#include "OperatorBinomialDistribution.h"

#include<DataNumber.h>
OperatorBinomialDistribution::OperatorBinomialDistribution()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Catalog");
    this->path().push_back("Discrete");
    this->setKey("PopulationOperatorBinomialDistribution");
    this->setName("binomial");
    this->setInformation("f(k)=$p^k(1-p)^\\{n-k\\}$ for 0<=k<n, 0 otherwise\n");
    this->structurePlug().addPlugIn(DataNumber::KEY,"p.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"n.num");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"f.dist");
}

void OperatorBinomialDistribution::exec(){
    double p = dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    double n = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    DistributionBinomial gen(p,n);
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(gen);

}

COperator * OperatorBinomialDistribution::clone(){
    return new OperatorBinomialDistribution();
}
