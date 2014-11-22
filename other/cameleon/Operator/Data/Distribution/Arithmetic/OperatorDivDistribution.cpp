#include "OperatorDivDistribution.h"
#include<DataDistribution.h>
#include"data/distribution/Distribution.h"
OperatorDivDistribution::OperatorDivDistribution()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Arithmetic");
    this->setKey("PopulationOperatorDivDistribution");
    this->setName("division");
    this->setInformation("h=f/g\n");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"g.dist");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"h.dist");
}

void OperatorDivDistribution::exec(){
    Distribution f= dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();
    Distribution g= dynamic_cast<DataDistribution *>(this->plugIn()[1]->getData())->getValue();

    f= f/g;
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(f);

}
COperator * OperatorDivDistribution::clone(){
    return new OperatorDivDistribution();
}
