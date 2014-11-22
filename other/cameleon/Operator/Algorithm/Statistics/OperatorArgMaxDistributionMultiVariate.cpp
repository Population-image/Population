#include "OperatorArgMaxDistributionMultiVariate.h"

#include"algorithm/Statistics.h"
#include<DataDistributionMultiVariate.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorArgMaxDistributionMultiVariate::OperatorArgMaxDistributionMultiVariate()
    :COperator()
{
        this->path().push_back("Algorithm");
    this->path().push_back("Statistics");
        this->path().push_back("DistributionMultiVariate");
    this->setKey("PopulationOperatorArgMaxDistributionMultiVariate");
    this->setName("argMax");

    this->setInformation("x=$\\mbox{arg}\\max_\\{\\forall x \\in R\\}f(x)$n");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataPoint::KEY,"xmin.num");
    this->structurePlug().addPlugIn(DataPoint::KEY,"xmax.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"step.num");
    this->structurePlug().addPlugOut(DataPoint::KEY,"h.dist");
}
void OperatorArgMaxDistributionMultiVariate::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugIn()[2]->setState(CPlug::EMPTY);
    if(this->plugIn()[3]->isConnected()==false)
        this->plugIn()[3]->setState(CPlug::OLD);
    else
        this->plugIn()[3]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorArgMaxDistributionMultiVariate::exec(){
    DistributionMultiVariate f= dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[0]->getData())->getValue();

    VecF64 xmin = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
    VecF64 xmax = dynamic_cast<DataPoint *>(this->plugIn()[2]->getData())->getValue();

    double step;
    if(this->plugIn()[3]->isDataAvailable()==true){
        step = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();
        if(step<=0)
            this->error("step must be superior to 0");
    }
    else{

        step = 0.01;

    }
    VecF64 value =  Statistics::argMax( f, xmin, xmax, step);
    dynamic_cast<DataPoint *>(this->plugOut()[0]->getData())->setValue(value);
}

COperator * OperatorArgMaxDistributionMultiVariate::clone(){
    return new OperatorArgMaxDistributionMultiVariate();
}
