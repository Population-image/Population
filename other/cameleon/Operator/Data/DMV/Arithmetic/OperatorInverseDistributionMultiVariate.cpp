#include "OperatorInverseDistributionMultiVariate.h"
#include"data/distribution/DistributionMultiVariateFromDataStructure.h"
#include<DataDistributionMultiVariate.h>
OperatorInverseDistributionMultiVariate::OperatorInverseDistributionMultiVariate()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("DistributionMultiVariate");
    this->path().push_back("Arithmetic");
    this->setKey("PopulationOperatorInverseDistributionMultiVariate");
    this->setName("inverse");
    this->setInformation("h=1/f\n");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"f.dist");
    this->structurePlug().addPlugOut(DataDistributionMultiVariate::KEY,"h.dist");
}

void OperatorInverseDistributionMultiVariate::exec(){
    DistributionMultiVariate f= dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[0]->getData())->getValue();

    DistributionMultiVariateExpression reg;
    reg.fromRegularExpression("1","x");
    DistributionMultiVariate gen(reg);
    f= gen/f;
    dynamic_cast<DataDistributionMultiVariate *>(this->plugOut()[0]->getData())->setValue(f);
}

COperator * OperatorInverseDistributionMultiVariate::clone(){
    return new OperatorInverseDistributionMultiVariate();
}
