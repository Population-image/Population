#include "OperatorComputedStaticticsDistribution.h"

#include<DataDistribution.h>
#include<DataPoint.h>
#include<DataNumber.h>
#include"algorithm/Statistics.h"
OperatorComputedStaticticsDistribution::OperatorComputedStaticticsDistribution()
    :COperator()
{

    this->path().push_back("Algorithm");
this->path().push_back("Statistics");
        this->path().push_back("Distribution");
    this->setKey("OperatorComputedStaticticsDistribution");
    this->setName("computedStaticticsFromRealRealizations");
    this->setInformation("$P(x=vmin+i*step)=\\frac{\\sum_j 1_{vmin+i*step<V(j)<vmin+i*(step+1)}}{\\sum_j 1_{V(j)==i}}$ discrete statistical quantities computed from realizations ");
    this->structurePlug().addPlugIn(DataPoint::KEY,"V.v");
    this->structurePlug().addPlugIn(DataNumber::KEY,"percentage.num");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"P.dist");
}
void OperatorComputedStaticticsDistribution::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorComputedStaticticsDistribution::exec(){

    VecF64  v = dynamic_cast<DataPoint *>(this->plugIn()[0]->getData())->getValue();
    double step=0.05;
    if(this->plugIn()[1]->isDataAvailable()==true)
        step = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    try{

        Distribution  dist = Statistics::computedStaticticsFromRealRealizations(v,step);
        dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(dist);
    }
    catch(pexception msg){
        this->error(msg.what());
    }

}

COperator * OperatorComputedStaticticsDistribution::clone(){
    return new OperatorComputedStaticticsDistribution();
}
