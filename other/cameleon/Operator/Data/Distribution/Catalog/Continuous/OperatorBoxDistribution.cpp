#include "OperatorBoxDistribution.h"
#include<DataNumber.h>
OperatorBoxDistribution::OperatorBoxDistribution()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Catalog");
    this->path().push_back("Continuous");
    this->setKey("PopulationOperatorBoxDistribution");
    this->setName("rectangular");
    this->setInformation("f(x)=1 for min<=x<=max;0 otherwise\n");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmin.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmax.num");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"f.dist");
}

void OperatorBoxDistribution::exec(){
    double xmin = dynamic_cast<DataNumber *>(this->plugIn()[0]->getData())->getValue();
    double xmax = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    DistributionRectangular gen(xmin,xmax);
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(gen);

}

COperator * OperatorBoxDistribution::clone(){
    return new OperatorBoxDistribution();
}
