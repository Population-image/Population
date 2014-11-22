#include "OperatorMaxDistribution.h"
#include<DataDistribution.h>
using namespace pop;
OperatorMaxDistribution::OperatorMaxDistribution()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Arithmetic");
    this->setKey("PopulationOperatorMaxDistribution");
    this->setName("maximum");
    this->setInformation("h=max(f,g)\n");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"g.dist");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"h.dist");
}

void OperatorMaxDistribution::exec(){
    Distribution f= dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();
    Distribution g= dynamic_cast<DataDistribution *>(this->plugIn()[1]->getData())->getValue();

    f= max(f,g);
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(f);
}

COperator * OperatorMaxDistribution::clone(){
    return new OperatorMaxDistribution();
}
