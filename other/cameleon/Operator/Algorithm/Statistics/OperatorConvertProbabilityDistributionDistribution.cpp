#include "OperatorConvertProbabilityDistributionDistribution.h"

#include"algorithm/Statistics.h"
#include<DataDistribution.h>
#include<DataNumber.h>
OperatorConvertProbabilityDistributionDistribution::OperatorConvertProbabilityDistributionDistribution()
    :COperator()
{

    this->path().push_back("Algorithm");
this->path().push_back("Statistics");
        this->path().push_back("Distribution");
    this->setKey("PopulationOperatorConvertProbabilityDistributionDistribution");
    this->setName("toProbabilityDistribution");
    this->setInformation("ConvertProbabilityDistributio h(x)=|f(x)|/norm for xmin<=x<xmax, 0 otherwise with norm =$\\int_\\{xmin\\}^\\{xmax\\} |f(x)|dx$");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmin.num(default xmin value of the distribution)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmax.num(default xmax value of the distribution)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"step.num(default step value of the distribution usually 0.01)");
    this->structurePlug().addPlugOut(DataDistribution::KEY,"h.dist");
}
void OperatorConvertProbabilityDistributionDistribution::initState(){
    this->plugIn()[0]->setState(CPlug::EMPTY);
    if(this->plugIn()[1]->isConnected()==false)
        this->plugIn()[1]->setState(CPlug::OLD);
    else
        this->plugIn()[1]->setState(CPlug::EMPTY);

    if(this->plugIn()[2]->isConnected()==false)
        this->plugIn()[2]->setState(CPlug::OLD);
    else
        this->plugIn()[2]->setState(CPlug::EMPTY);


    if(this->plugIn()[3]->isConnected()==false)
        this->plugIn()[3]->setState(CPlug::OLD);
    else
        this->plugIn()[3]->setState(CPlug::EMPTY);
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorConvertProbabilityDistributionDistribution::exec(){
    Distribution f= dynamic_cast<DataDistribution *>(this->plugIn()[0]->getData())->getValue();

    double xmin;
    if(this->plugIn()[1]->isDataAvailable()==true){
       xmin  = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    }else{
        xmin = f.getXmin();
        if(xmin==-numeric_limits<double>::max()){
            this->error("No xmin value for this distribution, you must set it manually");
        }
    }

    double xmax;
    if(this->plugIn()[2]->isDataAvailable()==true){
       xmax  = dynamic_cast<DataNumber *>(this->plugIn()[2]->getData())->getValue();
    }else{
        xmax = f.getXmax();
        if(xmax== numeric_limits<double>::max()){
            this->error("No xmax value for this distribution, you must set it manually");
        }
    }
    double step;
    if(this->plugIn()[3]->isDataAvailable()==true){
       step  = dynamic_cast<DataNumber *>(this->plugIn()[3]->getData())->getValue();
    }else{
        step = f.getStep();
    }

    Distribution dist= Distribution(Statistics::toProbabilityDistribution(f,xmin,xmax,step));
    dynamic_cast<DataDistribution *>(this->plugOut()[0]->getData())->setValue(dist);

}

COperator * OperatorConvertProbabilityDistributionDistribution::clone(){
    return new OperatorConvertProbabilityDistributionDistribution();
}
