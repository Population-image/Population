#include "OperatorMomentDistributionMultiVariate.h"

#include"algorithm/Statistics.h"
#include<DataDistribution.h>
#include<DataNumber.h>
#include<DataPoint.h>
OperatorMomentDistributionMultiVariate::OperatorMomentDistributionMultiVariate()
    :COperator()
{
    this->path().push_back("Algorithm");
    this->path().push_back("Statistics");
            this->path().push_back("DistributionMultiVariate");
    this->setKey("PopulationOperatorMomentDistributionMultiVariate");
    this->setName("moment");
    this->setInformation("$E(X^n) = \\int_\\{xmin\\}^\\{xmax\\} f(x)\\cdot x^m dx/ \\int_\\{xmin\\}^\\{xmax\\} f(x) dx $, for m=1, we have the expected value ");
    this->structurePlug().addPlugIn(DataDistributionMultiVariate::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataPoint::KEY,"m.num");
    this->structurePlug().addPlugIn(DataPoint::KEY,"xmin.num");
    this->structurePlug().addPlugIn(DataPoint::KEY,"xmax.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"step.num(by default 0.01)");
    this->structurePlug().addPlugOut(DataNumber::KEY,"E(X$^n$).num");
}
void OperatorMomentDistributionMultiVariate::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugIn()[2]->setState(CPlug::EMPTY);
    this->plugIn()[3]->setState(CPlug::EMPTY);
    if(this->plugIn()[4]->isConnected()==false)
        this->plugIn()[4]->setState(CPlug::OLD);
    else
        this->plugIn()[4]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorMomentDistributionMultiVariate::exec(){
    DistributionMultiVariate f= dynamic_cast<DataDistributionMultiVariate *>(this->plugIn()[0]->getData())->getValue();
    VecF64 m = dynamic_cast<DataPoint *>(this->plugIn()[1]->getData())->getValue();
    VecF64 xmin = dynamic_cast<DataPoint *>(this->plugIn()[2]->getData())->getValue();
    VecF64 xmax = dynamic_cast<DataPoint *>(this->plugIn()[3]->getData())->getValue();

    double step;
    if(this->plugIn()[4]->isDataAvailable()==true){
        step = dynamic_cast<DataNumber *>(this->plugIn()[4]->getData())->getValue();
        if(step<=0)
            this->error("step must be superior to 0");
    }
    else{
        step = 0.01;

    }
    double mom =  Statistics::moment(f,m, xmin, xmax, step);
    dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(mom);
}

COperator * OperatorMomentDistributionMultiVariate::clone(){
    return new OperatorMomentDistributionMultiVariate();
}
