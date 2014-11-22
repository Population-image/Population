#include "OperatorInnerProductDistribution.h"

#include"algorithm/Statistics.h"
#include<DataDistribution.h>
#include<DataNumber.h>
OperatorInnerProductDistribution::OperatorInnerProductDistribution()
    :COperator()
{

    this->path().push_back("Algorithm");
this->path().push_back("Statistics");
        this->path().push_back("Distribution");
    this->setKey("PopulationOperatorInnerProductDistribution");
    this->setName("productInner");
    this->setInformation("<f|g> = $\\int_\\{xmin\\}^\\{xmax\\} f(x)*g(x) dx\\$ ");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"g.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmin.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmax.num");
    this->structurePlug().addPlugIn(DataNumber::KEY,"step.num");
    this->structurePlug().addPlugOut(DataNumber::KEY,"E(X^n).num");
}
void OperatorInnerProductDistribution::initState(){
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
void OperatorInnerProductDistribution::exec(){
 Distribution f= dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();
 Distribution g= dynamic_cast<DataDistribution *>(this->plugIn()[1]->getData())->getValue();
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
    double InnerProduct =  Statistics::productInner(f,g, xmin, xmax, step);
    dynamic_cast<DataNumber *>(this->plugOut()[0]->getData())->setValue(InnerProduct);
}

COperator * OperatorInnerProductDistribution::clone(){
    return new OperatorInnerProductDistribution();
}
