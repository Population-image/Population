#include "OperatorCompoDistribution.h"

#include<DataDistribution.h>
OperatorCompositionDistribution::OperatorCompositionDistribution()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Arithmetic");
    this->setKey("PopulationOperatorCompositionDistribution");
    this->setName("composition");
    this->setInformation("h=f rho g\n");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"g.dist");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"h.dist");
}

void OperatorCompositionDistribution::exec(){
    Distribution f= dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();
    Distribution g= dynamic_cast<DataDistribution *>(this->plugIn()[1]->getData())->getValue();
    f.rho(g);
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(f);

}

COperator * OperatorCompositionDistribution::clone(){
    return new OperatorCompositionDistribution();
}
