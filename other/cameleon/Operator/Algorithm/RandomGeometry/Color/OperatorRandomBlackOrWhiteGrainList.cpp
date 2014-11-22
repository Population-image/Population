#include "OperatorRandomBlackOrWhiteGrainList.h"

#include <DataPoint.h>
#include <DataNumber.h>
#include <DataGrainList.h>
#include"algorithm/RandomGeometry.h"
using namespace pop;
OperatorRandomBlackOrWhiteGermGrain::OperatorRandomBlackOrWhiteGermGrain()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("Color");
    this->setKey("OperatorRandomBlackOrWhiteGrainList");
    this->setName("colorRandomBlackOrWhite");
    this->setInformation("phi'=phi such that each grain take a random black or white color");
    this->structurePlug().addPlugIn(DataGermGrain::KEY,"phi.grainlist");
    this->structurePlug().addPlugOut(DataGermGrain::KEY,"phi'.grainlist");
}


void OperatorRandomBlackOrWhiteGermGrain::exec(){

    shared_ptr<GermGrainMother> phi  = dynamic_cast<DataGermGrain *>(this->plugIn()[0]->getData())->getData() ;

    if(phi->dim==2)
    {
        GermGrain2 * phiin = dynamic_cast<GermGrain2 * >(phi.get());
        foo f;
        f(phiin);
        dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
    }
    else if (phi->dim==3)
    {
        GermGrain3 * phiin = dynamic_cast<GermGrain3 * >(phi.get());
        foo f;
        f(phiin);
        dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
    }
}

COperator * OperatorRandomBlackOrWhiteGermGrain::clone(){
    return new OperatorRandomBlackOrWhiteGermGrain;
}

