#include "OperatorCompoDistributionMultiVariate.h"

#include<DataDistributionMultiVariate.h>
OperatorCompositionDistributionMultiVariate::OperatorCompositionDistributionMultiVariate()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("DistributionMultiVariate");
    this->path().push_back("Arithmetic");
    this->setKey("PopulationOperatorCompositionDistributionMultiVariate");
    this->setName("composition");
    this->setInformation("h=f rho g\n");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"g.dist");
    this->structurePlug().addPlugOut(DataDistributionMultiVariate::KEY,"h.dist");
}

void OperatorCompositionDistributionMultiVariate::exec(){
    DistributionMultiVariate f= dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[0]->getData())->getValue();
    DistributionMultiVariate g= dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[1]->getData())->getValue();
    f.rho(g);
    dynamic_cast<DataDistributionMultiVariate *>(this->plugOut()[0]->getData())->setValue(f);

}

COperator * OperatorCompositionDistributionMultiVariate::clone(){
    return new OperatorCompositionDistributionMultiVariate();
}
