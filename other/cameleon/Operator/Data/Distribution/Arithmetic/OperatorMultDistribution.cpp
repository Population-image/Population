#include "OperatorMultDistribution.h"
#include<DataDistribution.h>
OperatorMultDistribution::OperatorMultDistribution()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Arithmetic");
    this->setKey("PopulationOperatorMultDistribution");
    this->setName("multiplication");
    this->setInformation("h=f*g\n");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"g.dist");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"h.dist");
}

void OperatorMultDistribution::exec(){
    Distribution f= dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();
    Distribution g= dynamic_cast<DataDistribution *>(this->plugIn()[1]->getData())->getValue();
    f= f*g;
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(f);
}

COperator * OperatorMultDistribution::clone(){
    return new OperatorMultDistribution();
}
