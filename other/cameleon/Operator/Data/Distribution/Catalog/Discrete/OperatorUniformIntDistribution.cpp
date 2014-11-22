#include "OperatorUniformIntDistribution.h"

#include<DataNumber.h>
OperatorUniformIntDistribution::OperatorUniformIntDistribution()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Catalog");
    this->path().push_back("Discrete");
    this->setKey("PopulationOperatorUniformIntDistribution");
    this->setName("uniformInt");
    this->setInformation("Uniform integer probability distribution in the range (xmin,xmax)\n");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmin.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmax.num");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"f.dist");
}

void OperatorUniformIntDistribution::exec(){
    int xmin = dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    int xmax = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    DistributionUniformInt gen (xmin,xmax);
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(gen);

}
COperator * OperatorUniformIntDistribution::clone(){
    return new OperatorUniformIntDistribution();
}
