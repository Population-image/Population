#include "OperatorRhombohedronGrainList.h"

#include <DataDistribution.h>
#include <DataDistributionMultiVariate.h>
#include <DataGrainList.h>
#include"algorithm/RandomGeometry.h"
using namespace pop;
OperatorRhombohedronGermGrain::OperatorRhombohedronGermGrain()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("Grain");
    this->setKey("OperatorRhombohedronGrainList");
    this->setName("rhombohedron");
    this->setInformation("phi'=$\\{C_0(x_0,R_0,A_0,\\Theta_0),...C_0(x_\\{n-1\\},R_\\{n-1\\},A_\\{n-1\\},\\Theta_\\{n-1\\}))$ where\n*phi=${x_0,...,x_\\{n-1}\\}$ is the input germ,\n* $R_i$ are 2 random variables for a 2D space (and 3 for a 3D space) following the probability distribution (P0,P1)\n*\n*Theta is 1D random variable for 2D space (without input, we sample a random angle in the unit sphere) and 3D random variable for a 3d space\n * C(x,R,A,theta) is a Rhombohedron centered in x of radius R, parallelepiped of angle A, and angle theta of orientation");
    this->structurePlug().addPlugIn(DataGermGrain::KEY,"phi.grainlist");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"R.dist");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"A.dist");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"Theta.dist");
    this->structurePlug().addPlugOut(DataGermGrain::KEY,"phi.grainlist");
}
void OperatorRhombohedronGermGrain::exec(){

    shared_ptr<GermGrainMother> phi  = dynamic_cast<DataGermGrain *>(this->plugIn()[0]->getData())->getData();
    Distribution dist1  = dynamic_cast<DataDistribution *>(this->plugIn()[1]->getData())->getValue();
    Distribution dist2  = dynamic_cast<DataDistribution *>(this->plugIn()[2]->getData())->getValue();
    DistributionMultiVariate angle =  dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[3]->getData())->getValue();
    try{
        if(GermGrain3 * phiin = dynamic_cast<GermGrain3 * >(phi.get()))
        {
            RandomGeometry::rhombohedron(*phiin,dist1 ,dist2,angle);
            dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
        }
    }
    catch(const pexception& e)
    {
        this->error(e.what());
    }

}

COperator * OperatorRhombohedronGermGrain::clone(){
    return new OperatorRhombohedronGermGrain;
}
