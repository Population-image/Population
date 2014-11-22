#include "OperatorRandomVariableDistribution.h"

#include"algorithm/Statistics.h"
#include<DataDistribution.h>
#include<DataNumber.h>
OperatorRandomVariableDistribution::OperatorRandomVariableDistribution()
    :COperator()
{

    this->path().push_back("Data");
    this->path().push_back("Distribution");
    this->path().push_back("Tool");
    this->setKey("PopulationOperatorRandomVariableDistribution");
    this->setName("randomVariable");
    this->setInformation("X a random variable following the probability distribution f");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugOut(DataNumber::KEY,"h.dist");
}

void OperatorRandomVariableDistribution::exec(){
    Distribution f= dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();
    try{
        double X =  f.randomVariable();
        dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(X);
    }catch(pexception msg){
        this->error(msg.what());
    }

}

COperator * OperatorRandomVariableDistribution::clone(){
    return new OperatorRandomVariableDistribution();
}
