#include "OperatorMaxValueDistributionMultiVariate.h"
#include"algorithm/Statistics.h"
#include<DataDistributionMultiVariate.h>
#include<DataPoint.h>
#include<DataNumber.h>
OperatorMaxValueDistributionMultiVariate::OperatorMaxValueDistributionMultiVariate()
    :COperator()
{
        this->path().push_back("Algorithm");
    this->path().push_back("Statistics");
            this->path().push_back("DistributionMultiVariate");
    this->setKey("PopulationOperatorMaxValueDistributionMultiVariate");
    this->setName("maxValue");

    this->setInformation("x=$ \\max_\\{ \\forall x \\in R\\}f(x)$");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataPoint::KEY,"xmin.num");
    this->structurePlug().addPlugIn(DataPoint::KEY,"xmax.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"step.num");
    this->structurePlug().addPlugOut(DataNumber::KEY,"min.num");
}
void OperatorMaxValueDistributionMultiVariate::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugIn()[2]->setState(CPlug::EMPTY);
    if(this->plugIn()[3]->isConnected()==false)
        this->plugIn()[3]->setState(CPlug::OLD);
    else
        this->plugIn()[3]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorMaxValueDistributionMultiVariate::exec(){
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
    pop::F64 value =  Statistics::maxValue( f, xmin, xmax, step);
    dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(value);
}

COperator * OperatorMaxValueDistributionMultiVariate::clone(){
    return new OperatorMaxValueDistributionMultiVariate();
}
