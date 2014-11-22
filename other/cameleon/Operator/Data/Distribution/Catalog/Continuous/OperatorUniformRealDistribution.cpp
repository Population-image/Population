#include "OperatorUniformRealDistribution.h"

#include<DataNumber.h>
OperatorUniformRealDistribution::OperatorUniformRealDistribution()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Catalog");
    this->path().push_back("Continuous");
    this->setKey("PopulationOperatorUniformRealDistribution");
    this->setName("uniformReal");
    this->setInformation("Uniform Real probability distribution in the range (xmin,xmax)\n");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmin.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmax.num");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"f.dist");
}

void OperatorUniformRealDistribution::exec(){
    double xmin = dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    double xmax = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    DistributionUniformReal gen (xmin,xmax);
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(gen);

}
COperator * OperatorUniformRealDistribution::clone(){
    return new OperatorUniformRealDistribution();
}
