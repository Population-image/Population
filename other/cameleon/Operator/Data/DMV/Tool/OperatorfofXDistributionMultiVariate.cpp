#include "OperatorfofXDistributionMultiVariate.h"
#include<DataDistributionMultiVariate.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorfofxDistributionMultiVariate::OperatorfofxDistributionMultiVariate()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("DistributionMultiVariate");
    this->path().push_back("Tool");
    this->setKey("PopulationOperatorfofxDistributionMultiVariate");
    this->setName("fofx");
    this->setInformation("y=f(x)");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"f.dist");

    this->structurePlug().addPlugIn(DataPoint::KEY,"x.vec");

    this->structurePlug().addPlugOut(DataNumber::KEY,"y.num");
}

void OperatorfofxDistributionMultiVariate::exec(){
    DistributionMultiVariate f= dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[0]->getData())->getValue();
    VecF64 value = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
    double y = f.operator ()(value);
    dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(y);
}

COperator * OperatorfofxDistributionMultiVariate::clone(){
    return new OperatorfofxDistributionMultiVariate();
}
