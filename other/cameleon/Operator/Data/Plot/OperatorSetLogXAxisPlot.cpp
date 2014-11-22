#include "OperatorSetLogXAxisPlot.h"

#include<CData.h>
#include<DataPlot.h>
#include<DataString.h>

#include "data/utility/PlotGraph.h"

OperatorSetLogXAxisPlot::OperatorSetLogXAxisPlot(){
        this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorSetLogXAxisPlot");
    this->setName("logXAxis");
    this->setInformation("set log scale of X-Axis");
    this->structurePlug().addPlugIn(DataPlot::KEY,"g.plot");
    this->structurePlug().addPlugOut(DataPlot::KEY,"f.plot");
}
void OperatorSetLogXAxisPlot::exec(){
    shared_ptr<PlotGraph> graph = dynamic_cast<DataPlot *>(this->plugIn()[0]->getData())->getData();


    graph->setXAxisLog(true);

    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(graph);
}
COperator * OperatorSetLogXAxisPlot::clone(){
    return new OperatorSetLogXAxisPlot();
}

