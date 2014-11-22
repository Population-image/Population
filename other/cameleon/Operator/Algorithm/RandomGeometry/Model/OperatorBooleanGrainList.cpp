#include "OperatorBooleanGrainList.h"

#include <DataDistribution.h>
#include <DataGrainList.h>
#include"algorithm/RandomGeometry.h"
using namespace pop;
OperatorBooleanGermGrain::OperatorBooleanGermGrain()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("Model");
    this->setKey("OperatorBooleanGrainList");
    this->setName("boolean");
    this->setInformation("phi'=ph where the model of phi' is boolean\n");
    this->structurePlug().addPlugIn(DataGermGrain::KEY,"phi.grainlist");
    this->structurePlug().addPlugOut(DataGermGrain::KEY,"phi'.grainlist");
}


void OperatorBooleanGermGrain::exec(){

    shared_ptr<GermGrainMother> phi  = dynamic_cast<DataGermGrain *>(this->plugIn()[0]->getData())->getData();
    phi->setModel(Boolean);
    dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);

}

COperator * OperatorBooleanGermGrain::clone(){
    return new OperatorBooleanGermGrain;
}
