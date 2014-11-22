#include "OperatorBlankPlot.h"

#include<CData.h>
#include<DataPlot.h>
#include<DataNumber.h>

#include "data/utility/PlotGraph.h"


OperatorBlankPlot::OperatorBlankPlot(){
    _onetime = true;
        this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorBlankPlot");
    this->setName("blank");
    this->setInformation("Create a blank graph");
    this->structurePlug().addPlugOut(DataPlot::KEY,"g.plot");
}
void OperatorBlankPlot::exec(){
    _onetime = false;
    PlotSingleGraph graph;
    shared_ptr<PlotGraph> m(new PlotGraph(graph));
    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(m);
}
COperator * OperatorBlankPlot::clone(){
    return new OperatorBlankPlot();
}
void OperatorBlankPlot::initState(){
    COperator::initState();
    _onetime = true;

}
bool OperatorBlankPlot::executionCondition(){
    if(_onetime==true)
        return COperator::executionCondition();
    else
        return false;
}
