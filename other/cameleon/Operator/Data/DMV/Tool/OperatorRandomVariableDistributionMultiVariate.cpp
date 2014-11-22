#include "OperatorRandomVariableDistributionMultiVariate.h"

#include"algorithm/Statistics.h"
#include<DataDistributionMultiVariate.h>
#include<DataPoint.h>
OperatorRandomVariableDistributionMultiVariate::OperatorRandomVariableDistributionMultiVariate()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("DistributionMultiVariate");
    this->path().push_back("Tool");
    this->setKey("PopulationOperatorRandomVariableDistributionMultiVariate");
    this->setName("randomVariable");
    this->setInformation("X a random multivariate random variable following the probability DistributionMultiVariate f");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"f.dist");
    this->structurePlug().addPlugOut(DataPoint::KEY,"X.vec");
}

void OperatorRandomVariableDistributionMultiVariate::exec(){
    DistributionMultiVariate f= dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[0]->getData())->getValue();
    try{
        VecF64 X =  f.randomVariable();
        dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(X);
    }catch(pexception msg){
        this->error(msg.what());
    }

}

COperator * OperatorRandomVariableDistributionMultiVariate::clone(){
    return new OperatorRandomVariableDistributionMultiVariate();
}
