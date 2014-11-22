#include "OperatorArgMinDistribution.h"
#include"algorithm/Statistics.h"
#include<DataDistribution.h>
#include<DataNumber.h>
OperatorArgMinDistribution::OperatorArgMinDistribution()
    :COperator()
{
        this->path().push_back("Algorithm");
    this->path().push_back("Statistics");
        this->path().push_back("Distribution");
    this->setKey("PopulationOperatorArgMinDistribution");
    this->setName("argMin");

    this->setInformation("x=$\\mbox{arg}\\min_\\{\\forall x \\in R\\}f(x)$n");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmin.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmax.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"step.num");
    this->structurePlug().addPlugOut(DataNumber::KEY,"h.dist");
}
void OperatorArgMinDistribution::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    this->plugIn()[1]->setState(CPlug::EMPTY);
    this->plugIn()[2]->setState(CPlug::EMPTY);
    if(this->plugIn()[3]->isConnected()==false)
        this->plugIn()[3]->setState(CPlug::OLD);
    else
        this->plugIn()[3]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorArgMinDistribution::exec(){
    Distribution f= dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();

    double xmin = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    double xmax = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();

    double step;
    if(this->plugIn()[3]->isDataAvailable()==true){
        step = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();
        if(step<=0)
            this->error("step must be superior to 0");
    }
    else{

        step = 0.01;

    }
    double value =  Statistics::argMin( f, xmin, xmax, step);
    dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(value);
}

COperator * OperatorArgMinDistribution::clone(){
    return new OperatorArgMinDistribution();
}
