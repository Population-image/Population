#include "OperatorRandomColorGrainList.h"

#include <DataDistribution.h>
#include <DataDistribution.h>
#include <DataDistribution.h>
#include <DataGrainList.h>
#include"algorithm/RandomGeometry.h"
using namespace pop;
OperatorRandomColorGermGrain::OperatorRandomColorGermGrain()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("Color");
    this->setKey("OperatorRandomColorGrainList");
    this->setName("colorRandom");
    this->setInformation("phi'=phi such that each grain take a random color (R,G,B) following the probabilitu distribution Pr, Pg, Pb");
    this->structurePlug().addPlugIn(DataGermGrain::KEY,"phi.grainlist");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"Pr.dist");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"Pg.dist");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"Pb.dist");
    this->structurePlug().addPlugOut(DataGermGrain::KEY,"phi'.grainlist");
}
void OperatorRandomColorGermGrain::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);

    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);

    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);

    if(this->plugIn()[3]->isConnected()==false)
        this->plugIn()[3]->setState(CPlug::OLD);
    else
        this->plugIn()[3]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}

void OperatorRandomColorGermGrain::exec(){

    shared_ptr<GermGrainMother> phi  = dynamic_cast<DataGermGrain *>(this->plugIn()[0]->getData())->getData() ;

    shared_ptr<Distribution> Pr;
    shared_ptr<Distribution> Pg;
    shared_ptr<Distribution> Pb;
    if(this->plugIn()[1]->isDataAvailable()==true){
        Pr = dynamic_cast<DataDistribution *>(this->plugIn()[1]->getData())->getData();
    }
    else
    {
        Pr = shared_ptr<Distribution>(new DistributionUniformInt(0,255));
    }
    if(this->plugIn()[2]->isDataAvailable()==true){
        Pg = dynamic_cast<DataDistribution *>(this->plugIn()[2]->getData())->getData();
    }
    else
    {
        Pg = shared_ptr<Distribution>(new DistributionUniformInt(0,255));
    }
    if(this->plugIn()[3]->isDataAvailable()==true){
        Pb = dynamic_cast<DataDistribution *>(this->plugIn()[3]->getData())->getData();
    }
    else
    {
        Pb = shared_ptr<Distribution>(new DistributionUniformInt(0,255));
    }
    Distribution * Prc=Pr.get();
    Distribution * Pgc=Pg.get();
    Distribution * Pbc=Pb.get();
    if(phi->dim==2)
    {
        GermGrain2 * phiin = dynamic_cast<GermGrain2 * >(phi.get());
        GermGrain2 * phiout;
        foo f;
        f(phiin, Prc, Pgc, Pbc);
        dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
    }
    else if (phi->dim==3)
    {

        GermGrain3 * phiin = dynamic_cast<GermGrain3 * >(phi.get());
        foo f;
        f(phiin, Prc, Pgc, Pbc);
        dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
    }
}

COperator * OperatorRandomColorGermGrain::clone(){
    return new OperatorRandomColorGermGrain;
}
