#include "OperatorNormalDistribution.h"

#include<DataNumber.h>

OperatorNormalDistribution::OperatorNormalDistribution()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Catalog");
    this->path().push_back("Continuous");
    this->setKey("PopulationOperatorNormalDistribution");
    this->setName("normal");
    this->setInformation("f(x)=$\\frac\\{1\\}\\{\\sqrt\\{2 \\pi \\sigma^2\\}\\} e^\\{ - \\frac\\{(x-mean)^2\\}\\{2 \\sigma^2\\}\\}$");
    this->structurePlug().addPlugIn(DataNumber::KEY,"mean.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"sigma.num");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"f.dist");
}

void OperatorNormalDistribution::exec(){
    double mean = dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    double standart_deviation = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    DistributionNormal gen(mean,standart_deviation);

    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(gen);

}

COperator * OperatorNormalDistribution::clone(){
    return new OperatorNormalDistribution();
}
