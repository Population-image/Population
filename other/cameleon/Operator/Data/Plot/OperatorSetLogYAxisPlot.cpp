#include "OperatorSetLogYAxisPlot.h"

#include<CData.h>
#include<DataPlot.h>
#include<DataString.h>

#include "data/utility/PlotGraph.h"

OperatorSetLogYAxisPlot::OperatorSetLogYAxisPlot(){
        this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorSetLogYAxisPlot");
    this->setName("logYAxis");
    this->setInformation("set log scale of X-Axis");
    this->structurePlug().addPlugIn(DataPlot::KEY,"g.plot");
    this->structurePlug().addPlugOut(DataPlot::KEY,"f.plot");
}
void OperatorSetLogYAxisPlot::exec(){
    shared_ptr<PlotGraph> graph = dynamic_cast<DataPlot *>(this->plugIn()[0]->getData())->getData();


    graph->setYAxisLog(true);

    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(graph);
}
COperator * OperatorSetLogYAxisPlot::clone(){
    return new OperatorSetLogYAxisPlot();
}
