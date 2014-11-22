#include "OperatorPolyhedraGrainList.h"

#include <DataDistribution.h>

#include <DataVector.h>
#include <DataGrainList.h>
#include"algorithm/RandomGeometry.h"
using namespace pop;
OperatorPolyhedraGermGrain::OperatorPolyhedraGermGrain()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("Grain");
    this->setKey("OperatorPolyhedraGrainList");
    this->setName("polyhedra");
    this->setInformation("phi'=$\\{C_0(x_0,r_{0,0},n_{0,0}...,r_{0,p},n_{0,p},\\Theta_0),...C_0(x_\\{n-1\\},,r_\\{n-1,0\\},n_\\{n-1,0\\}...,r_\\{n-1,p\\},n_\\{n-1,p\\},\\Theta_\\{n-1\\}))$ where\n*phi'=$\\{x_0,...,x_\\{n-1\\}\\}$ is the input germ,\n* $R_i$ are 2 random variables for a 2D space (and 3 for a 3D space) following the probability distribution (P0,P1)\n*\n*Theta is 1D random variable for 2D space (without input, we sample a random angle in the unit sphere) and 3D random variable for a 3d space\n * C(x,R,A,theta) is a Polyhedra centered in x delimited by p plane at a given radius and normal, and angle theta of orientation");
    this->structurePlug().addPlugIn(DataGermGrain::KEY,"phi.grainlist");
    this->structurePlug().addPlugIn(DataVector::KEY,"vradius.dist");
    this->structurePlug().addPlugIn(DataVector::KEY,"vnorma.dist");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"Thetax.dist");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"Thetay.dist");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"Thetaz.dist");
    this->structurePlug().addPlugOut(DataGermGrain::KEY,"phi'.grainlist");
}

void OperatorPolyhedraGermGrain::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugIn()[2]->setState(CPlug::EMPTY);

    if(this->plugIn()[3]->isConnected()==false)
        this->plugIn()[3]->setState(CPlug::OLD);
    else
        this->plugIn()[3]->setState(CPlug::EMPTY);

    if(this->plugIn()[4]->isConnected()==false)
        this->plugIn()[4]->setState(CPlug::OLD);
    else
        this->plugIn()[4]->setState(CPlug::EMPTY);

    if(this->plugIn()[5]->isConnected()==false)
        this->plugIn()[5]->setState(CPlug::OLD);
    else
        this->plugIn()[5]->setState(CPlug::EMPTY);


    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorPolyhedraGermGrain::exec(){

    shared_ptr<GermGrainMother> phi  = dynamic_cast<DataGermGrain *>(this->plugIn()[0]->getData())->getData();

    DataVector::DATAVECTOR v_dist1  = dynamic_cast<DataVector  *>(this->plugIn()[1]->getData())->getData();
    vector<Distribution *> v_dist1c;
    for(int i =0;i<(int)v_dist1->size();i++){
        v_dist1c.push_back(dynamic_cast<Distribution *>(v_dist1->operator [](i).get()));
    }

    DataVector::DATAVECTOR v_dist2  = dynamic_cast<DataVector  *>(this->plugIn()[1]->getData())->getData();
    vector<Distribution *> v_dist2c;
    for(int i =0;i<(int)v_dist2->size();i++){
        v_dist2c.push_back(dynamic_cast<Distribution *>(v_dist2->operator [](i).get()));
    }
    shared_ptr<Distribution> distangle1;
    shared_ptr<Distribution> distangle2;
    shared_ptr<Distribution> distangle3;
    if(phi->dim==2)
    {
        Distribution * angle;
        if(this->plugIn()[3]->isDataAvailable()==true){
            distangle1 = dynamic_cast<DataDistribution *>(this->plugIn()[3]->getData())->getData();
        }
        else
        {
            distangle1 = shared_ptr<Distribution>(new DistributionUniformReal(0,360));
        }
        angle =distangle1.get();
        GermGrain2 * phiin = dynamic_cast<GermGrain2 * >(phi.get());

        foo f;
        f(phiin,v_dist1c,v_dist2c,angle);
        dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
    }
    else if (phi->dim==3)
    {

        if(this->plugIn()[3]->isDataAvailable()==true){
            distangle1 = dynamic_cast<DataDistribution *>(this->plugIn()[3]->getData())->getData();
        }
        else
        {
            distangle1 = shared_ptr<Distribution>(new DistributionUniformReal(0,360));
        }
        if(this->plugIn()[4]->isDataAvailable()==true){
            distangle2 = dynamic_cast<DataDistribution *>(this->plugIn()[4]->getData())->getData();
        }
        else
        {
            distangle2 = shared_ptr<Distribution>(new DistributionUniformReal(0,360));
        }
        if(this->plugIn()[5]->isDataAvailable()==true){
            distangle3 = dynamic_cast<DataDistribution *>(this->plugIn()[5]->getData())->getData();
        }
        else
        {
            distangle3 = shared_ptr<Distribution>(new DistributionUniformReal(0,360));
        }
        Distribution * anglex =distangle1.get();
        Distribution * angley =distangle2.get();
        Distribution * anglez =distangle3.get();
        GermGrain3 * phiin = dynamic_cast<GermGrain3 * >(phi.get());
        foo f;
        f(phiin,v_dist1c,v_dist2c,anglex,angley,anglez);
        dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(phi);
    }
    else{
        this->error("In 3d dimension, the input Rx, Ry and Rz must be set!");
    }

}
COperator * OperatorPolyhedraGermGrain::clone(){
    return new OperatorPolyhedraGermGrain;
}
