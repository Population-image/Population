#include "OperatorSubDistributionMultiVariate.h"

#include<DataDistributionMultiVariate.h>
OperatorSubDistributionMultiVariate::OperatorSubDistributionMultiVariate()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("DistributionMultiVariate");
    this->path().push_back("Arithmetic");
    this->setKey("PopulationOperatorSubDistributionMultiVariate");
    this->setName("subtraction");
    this->setInformation("h=f-g\n");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"g.dist");
    this->structurePlug().addPlugOut(DataDistributionMultiVariate::KEY,"h.dist");
}

void OperatorSubDistributionMultiVariate::exec(){
    DistributionMultiVariate f= dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[0]->getData())->getValue();
    DistributionMultiVariate g= dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[1]->getData())->getValue();

    f= f-g;
    dynamic_cast<DataDistributionMultiVariate *>(this->plugOut()[0]->getData())->setValue(f);
}

COperator * OperatorSubDistributionMultiVariate::clone(){
    return new OperatorSubDistributionMultiVariate();
}
