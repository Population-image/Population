#include "OperatorMinOverlapGrainList.h"

#include <DataPoint.h>
#include <DataNumber.h>
#include <DataGrainList.h>
#include"algorithm/RandomGeometry.h"
using namespace pop;
OperatorMinOverlapGermGrain::OperatorMinOverlapGermGrain()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("Germ");
    this->setKey("OperatorMinOverlapGrainList");
    this->setName("minOverlapFilter");
    this->setInformation("phi'=f(phi,radius) where f is the min overlap filter\n");
    this->structurePlug().addPlugIn(DataGermGrain::KEY,"phi.grainlist");
    this->structurePlug().addPlugIn(DataNumber::KEY,"radius.dist");
    this->structurePlug().addPlugOut(DataGermGrain::KEY,"phi'.grainlist");
}


void OperatorMinOverlapGermGrain::exec(){

    shared_ptr<GermGrainMother> phi  = dynamic_cast<DataGermGrain *>(this->plugIn()[0]->getData())->getData();
    double radius = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();

    if(phi->dim==2)
    {
        GermGrain2 * phiin = dynamic_cast<GermGrain2 * >(phi.get());
        foo f;
        f(phiin,radius);
        dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
    }
    else if (phi->dim==3)
    {
        GermGrain3 * phiin = dynamic_cast<GermGrain3 * >(phi.get());
        foo f;
        f(phiin,radius);
        dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
    }
}

COperator * OperatorMinOverlapGermGrain::clone(){
    return new OperatorMinOverlapGermGrain;
}
