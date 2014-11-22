#include "OperatorSphereGrainList.h"

#include <DataDistribution.h>
#include <DataGrainList.h>
#include"algorithm/RandomGeometry.h"
using namespace pop;
OperatorSphereGermGrain::OperatorSphereGermGrain()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("Grain");
    this->setKey("OperatorSphereGrainList");
    this->setName("sphere");
    this->setInformation("phi'=$\\{S_0(x_0,R_0),...S_0(x_\\{n-1\\},R_\\{n-1\\}))$ where phi=$\\{x_0,...,x_\\{n-1\\}\\}$ is the input germ, $R_i$ are random variable following the probability distribution P and S(x,R) is a sphere centered in x of radius R \n");
    this->structurePlug().addPlugIn(DataGermGrain::KEY,"phi.grainlist");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"P.dist");
    this->structurePlug().addPlugOut(DataGermGrain::KEY,"phi.grainlist");
}


void OperatorSphereGermGrain::exec(){

    try{
    shared_ptr<GermGrainMother> phi  = dynamic_cast<DataGermGrain *>(this->plugIn()[0]->getData())->getData();
    shared_ptr<Distribution> dist  = dynamic_cast<DataDistribution *>(this->plugIn()[1]->getData())->getData();
    Distribution * distc = dist.get();
    if(phi->dim==2)
    {
        GermGrain2 * phiin = dynamic_cast<GermGrain2 * >(phi.get());

        foo f;
        f(phiin,distc);
        dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
    }
    else if (phi->dim==3)
    {
        GermGrain3 * phiin = dynamic_cast<GermGrain3 * >(phi.get());

        foo f;
        f(phiin,distc);
        dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
    }
    }
    catch(pexception msg){
        this->error(msg.what());
        return ;
    }
}

COperator * OperatorSphereGermGrain::clone(){
    return new OperatorSphereGermGrain;
}
