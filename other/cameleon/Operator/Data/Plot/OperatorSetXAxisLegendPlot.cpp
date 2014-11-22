#include "OperatorSetXAxisLegendPlot.h"

#include<CData.h>
#include<DataPlot.h>
#include<DataString.h>

#include "data/utility/PlotGraph.h"
OperatorSetXAxisPlot::OperatorSetXAxisPlot(){
        this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorSetXAxisPlot");
    this->setName("xAxisLegend");
    this->setInformation("set the XAxis legend of the graph");
    this->structurePlug().addPlugIn(DataPlot::KEY,"g.plot");
    this->structurePlug().addPlugIn(DataString::KEY,"XAxis.str");;
    this->structurePlug().addPlugOut(DataPlot::KEY,"f.plot");
}
void OperatorSetXAxisPlot::exec(){
    shared_ptr<PlotGraph> graph = dynamic_cast<DataPlot *>(this->plugIn()[0]->getData())->getData();

    string str = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    graph->setXAxixLegend(str);

    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(graph);
}
COperator * OperatorSetXAxisPlot::clone(){
    return new OperatorSetXAxisPlot();
}
