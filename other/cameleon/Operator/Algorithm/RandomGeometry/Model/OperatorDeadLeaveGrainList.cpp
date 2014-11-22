#include "OperatorDeadLeaveGrainList.h"
#include <DataDistribution.h>
#include <DataGrainList.h>
#include"algorithm/RandomGeometry.h"
using namespace pop;
OperatorDeadLeaveGermGrain::OperatorDeadLeaveGermGrain()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("Model");
    this->setKey("OperatorDeadLeaveGrainList");
    this->setName("deadLeave");
    this->setInformation("phi'=pho where the mdoel of phi' is dead leaves\n");
    this->structurePlug().addPlugIn(DataGermGrain::KEY,"phi.grainlist");
    this->structurePlug().addPlugOut(DataGermGrain::KEY,"phi'.grainlist");
}


void OperatorDeadLeaveGermGrain::exec(){

    shared_ptr<GermGrainMother> phi  = dynamic_cast<DataGermGrain *>(this->plugIn()[0]->getData())->getData();
    phi->setModel(DeadLeave);
    dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);

}

COperator * OperatorDeadLeaveGermGrain::clone(){
    return new OperatorDeadLeaveGermGrain;
}
