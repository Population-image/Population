#include "OperatorDistanceDistribution.h"

#include"algorithm/Statistics.h"
#include<DataDistribution.h>
#include<DataNumber.h>
OperatorDistanceDistribution::OperatorDistanceDistribution()
    :COperator()
{
    this->path().push_back("Algorithm");
this->path().push_back("Statistics");
        this->path().push_back("Distribution");
    this->setKey("PopulationOperatorDistanceDistribution");
    this->setName("norm");
    this->setInformation("$d(f)_n = (\\int_\\{xmin\\}^\\{xmax\\} |f(x)|^n dx)^(1/n)$, for m=1, we have the expected value ");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataNumber::KEY,"m.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmin.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmax.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"step.num");
    this->structurePlug().addPlugOut(DataNumber::KEY,"d(f)$_n$.num");
}
void OperatorDistanceDistribution::initState(){
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
void OperatorDistanceDistribution::exec(){
 Distribution f= dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();
    double m = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    double xmin = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    double xmax = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();

    double step;
    if(this->plugIn()[4]->isDataAvailable()==true){
        step = dynamic_cast<DataNumber *>(this->plugIn()[4]->getData())->getValue();
        if(step<=0)
            this->error("step must be superior to 0");
    }
    else{
            step = 0.01;

    }
    double Distance =  Statistics::norm(f,m, xmin, xmax, step);
    dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(Distance);
}

COperator * OperatorDistanceDistribution::clone(){
    return new OperatorDistanceDistribution();
}
