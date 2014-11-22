#include "OperatorfofXDistribution.h"
#include<DataDistribution.h>
#include<DataNumber.h>
OperatorfofxDistribution::OperatorfofxDistribution()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Tool");
    this->setKey("PopulationOperatorfofxDistribution");
    this->setName("fofx");
    this->setInformation("y=f(x)");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");

    this->structurePlug().addPlugIn(DataNumber::KEY,"x.num");

    this->structurePlug().addPlugOut(DataNumber::KEY,"y.num");
}

void OperatorfofxDistribution::exec(){
    Distribution f= dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();

    double value = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    double y = f.operator ()(value);
    dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(y);
}

COperator * OperatorfofxDistribution::clone(){
    return new OperatorfofxDistribution();
}
