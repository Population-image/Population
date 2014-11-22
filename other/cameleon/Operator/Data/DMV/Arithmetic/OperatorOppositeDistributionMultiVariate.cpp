#include "OperatorOppositeDistributionMultiVariate.h"
#include<DataDistributionMultiVariate.h>
#include"data/distribution/DistributionMultiVariateFromDataStructure.h"
OperatorOppositeDistributionMultiVariate::OperatorOppositeDistributionMultiVariate()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("DistributionMultiVariate");
    this->path().push_back("Arithmetic");
    this->setKey("PopulationOperatorOppositeDistributionMultiVariate");
    this->setName("oppoiste");
    this->setInformation("h=-f\n");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"f.dist");
    this->structurePlug().addPlugOut(DataDistributionMultiVariate::KEY,"h.dist");
}

void OperatorOppositeDistributionMultiVariate::exec(){
    DistributionMultiVariate f= dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[0]->getData())->getValue();

    DistributionMultiVariateExpression reg;
    reg.fromRegularExpression("1","x");
    DistributionMultiVariate gen(reg);
    f= gen -f;
    dynamic_cast<DataDistributionMultiVariate *>(this->plugOut()[0]->getData())->setValue(f);
}

COperator * OperatorOppositeDistributionMultiVariate::clone(){
    return new OperatorOppositeDistributionMultiVariate();
}
