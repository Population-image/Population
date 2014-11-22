#include "OperatorEllipsoidGrainList.h"

#include <DataDistribution.h>
#include <DataGrainList.h>
#include"algorithm/RandomGeometry.h"
#include"DataDistributionMultiVariate.h"
OperatorEllipsoidGermGrain::OperatorEllipsoidGermGrain()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("Grain");
    this->setKey("OperatorEllipsoidGrainList");
    this->setName("ellipsoid");
    this->setInformation("phi'=$\\{S_0(x_0,R_0,\\Theta_0),...S_0(x_\\{n-1\\},R_\\{n-1\\},\\Theta_\\{n-1\\}))$ where\n*phi=$\\{x_0,...,x_\\{n-1\\}\\}$ is the input germ,\n* $R_i$ are 2 random variables for a 2D space (and 3 for a 3D space) following the probability distribution (P0,P1)\n*\n*Theta is 1D random variable for 2D space (without input, we sample a random angle in the unit sphere) and 3D random variable for a 3d space\n *B(x,R,theta) is a Ellipsoid centered in x of radius R and angle theta");
    this->structurePlug().addPlugIn(DataGermGrain::KEY,"phi.grainlist");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"R.dist");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"Thetax.dist");
    this->structurePlug().addPlugOut(DataGermGrain::KEY,"phi.grainlist");
}

void OperatorEllipsoidGermGrain::exec(){

    shared_ptr<GermGrainMother> phi  = dynamic_cast<DataGermGrain *>(this->plugIn()[0]->getData())->getData();
    DistributionMultiVariate r =  dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[1]->getData())->getValue();
    DistributionMultiVariate angle =  dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[2]->getData())->getValue();
    try{
        if(GermGrain2 * phiin = dynamic_cast<GermGrain2 * >(phi.get())){

            RandomGeometry::ellipsoid(*phiin,r ,angle);
            dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
        }else if(GermGrain3 * phiin = dynamic_cast<GermGrain3 * >(phi.get()))
        {
            RandomGeometry::ellipsoid(*phiin,r ,angle);
            dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
        }
    }
    catch(const pexception& e)
    {
        this->error(e.what());
    }

}

COperator * OperatorEllipsoidGermGrain::clone(){
    return new OperatorEllipsoidGermGrain;
}
