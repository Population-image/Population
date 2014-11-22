#include "OperatorTransparencyGrainList.h"

#include <DataDistribution.h>
#include <DataGrainList.h>
#include <DataNumber.h>
#include"algorithm/RandomGeometry.h"
using namespace pop;
OperatorTransparencyGermGrain::OperatorTransparencyGermGrain()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("Model");
    this->setKey("OperatorTransparencyGrainList");
    this->setName("deadLeaveTransparent");
    this->setInformation("phi'=phi where the model of phi' is transparent dead leaves\n");
    this->structurePlug().addPlugIn(DataGermGrain::KEY,"phi.grainlist");
        this->structurePlug().addPlugIn(DataNumber::KEY,"alpha.num");
    this->structurePlug().addPlugOut(DataGermGrain::KEY,"phi'.grainlist");
}


void OperatorTransparencyGermGrain::exec(){

    shared_ptr<GermGrainMother> phi  = dynamic_cast<DataGermGrain *>(this->plugIn()[0]->getData())->getData();
    double t = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    phi->setModel(Transparent);
    phi->setTransparency(t);
    dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);

}

COperator * OperatorTransparencyGermGrain::clone(){
    return new OperatorTransparencyGermGrain;
}

