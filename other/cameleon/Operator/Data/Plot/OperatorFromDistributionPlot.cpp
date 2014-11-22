#include "OperatorFromDistributionPlot.h"

#include"algorithm/Statistics.h"
#include<DataDistribution.h>
#include<DataMatrix.h>
#include<DataNumber.h>
#include<DataPlot.h>
#include "data/utility/PlotGraph.h"

#include"algorithm/Statistics.h"
OperatorFromDistributionPlot::OperatorFromDistributionPlot()
    :COperator()
{
    this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("PopulationOperatorFromDistributionPlot");
    this->setName("fromDistribution");
    this->setInformation("A graph with these points: X(i)=xmin+i*step  and Y(i)=f(xmin+i*step)");
    this->structurePlug().addPlugIn(DataDistribution::KEY,"f.dist");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmin.num(default xmin value of the distribution)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"xmax.num(default xmax value of the distribution)");
    this->structurePlug().addPlugIn(DataNumber::KEY,"step.num(default step value of the distribution usually 0.01)");
    this->structurePlug().addPlugOut(DataPlot::KEY,"g.plot");
}
void OperatorFromDistributionPlot::initState(){
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
    this->plugOut()[0]->setState(CPlug::EMPTY);
}
void OperatorFromDistributionPlot::exec(){
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
    Mat2F64 m= Statistics::toMatrix(f, xmin, xmax, step);
    PlotGraph* graph(new PlotGraph());
    * graph = PlotGraphProcedureFromMatrix(m,0,1);
    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(shared_ptr<PlotGraph>(graph));
}

COperator * OperatorFromDistributionPlot::clone(){
    return new OperatorFromDistributionPlot();
}
