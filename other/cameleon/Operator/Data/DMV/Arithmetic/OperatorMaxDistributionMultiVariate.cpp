#include "OperatorMaxDistributionMultiVariate.h"
#include<DataDistributionMultiVariate.h>
using namespace pop;
OperatorMaxDistributionMultiVariate::OperatorMaxDistributionMultiVariate()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("DistributionMultiVariate");
    this->path().push_back("Arithmetic");
    this->setKey("PopulationOperatorMaxDistributionMultiVariate");
    this->setName("maximum");
    this->setInformation("h=max(f,g)\n");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"g.dist");
    this->structurePlug().addPlugOut(DataDistributionMultiVariate::KEY,"h.dist");
}

void OperatorMaxDistributionMultiVariate::exec(){
    DistributionMultiVariate f= dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[0]->getData())->getValue();
    DistributionMultiVariate g= dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[1]->getData())->getValue();

    f= max(f,g);
    dynamic_cast<DataDistributionMultiVariate *>(this->plugOut()[0]->getData())->setValue(f);
}

COperator * OperatorMaxDistributionMultiVariate::clone(){
    return new OperatorMaxDistributionMultiVariate();
}
