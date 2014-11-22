#include "OperatorRandomUniformPointGrainList.h"
#include <DataPoint.h>
#include <DataNumber.h>
#include <DataGrainList.h>
#include"algorithm/RandomGeometry.h"
using namespace pop;
OperatorRandomUniformPointGermGrain::OperatorRandomUniformPointGermGrain()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("RandomGeometry");
    this->path().push_back("Germ");
    this->setKey("OperatorRandomUniformPointGrainList");
    this->setName("poissonPointProcessUniform");
    this->setInformation("phi=$\\{x_0,...,x_\\{n-1\\}\\}$ where x are random variables thrown in the space D with intensity lmabda \n");
    this->structurePlug().addPlugIn(DataPoint::KEY,"D.v");
    this->structurePlug().addPlugIn(DataNumber::KEY,"lambda.dist");
    this->structurePlug().addPlugOut(DataGermGrain::KEY,"phi.grainlist");
}


void OperatorRandomUniformPointGermGrain::exec(){
    VecF64  v = dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();
    double lambda = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();

    if(v.size()==2){
        VecN<2,double> domain;
        domain(0)=v(0);
        domain(1)=v(1);
        GermGrain2  * f1 = new GermGrain2;
        * f1=   RandomGeometry::poissonPointProcess(domain,lambda);
        dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(shared_ptr<GermGrainMother>(f1));
    }
    else if(v.size()==3){
        VecN<3,double> domain;
        domain(0)=v(0);
        domain(1)=v(1);
        domain(2)=v(2);
        GermGrain3  * f1 = new GermGrain3;
        * f1 =   RandomGeometry::poissonPointProcess(domain,lambda);
        dynamic_cast<DataGermGrain *>(this->plugOut()[0]->getData())->setData(shared_ptr<GermGrainMother>(f1));
    }
    else{
        this->error("Space dimension must be 2 or 3");
    }
}

COperator * OperatorRandomUniformPointGermGrain::clone(){
    return new OperatorRandomUniformPointGermGrain;
}
