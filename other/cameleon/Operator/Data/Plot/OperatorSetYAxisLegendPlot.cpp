#include "OperatorSetYAxisLegendPlot.h"

#include<CData.h>
#include<DataPlot.h>
#include<DataString.h>

#include "data/utility/PlotGraph.h"
OperatorSetYAxisPlot::OperatorSetYAxisPlot(){
            this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorSetYAxisPlot");
    this->setName("yAxisLegend");
    this->setInformation("set the YAxis of the graph");
    this->structurePlug().addPlugIn(DataPlot::KEY,"g.plot");
    this->structurePlug().addPlugIn(DataString::KEY,"YAxis.str");;
    this->structurePlug().addPlugOut(DataPlot::KEY,"f.plot");
}
void OperatorSetYAxisPlot::exec(){
    shared_ptr<PlotGraph> graph = dynamic_cast<DataPlot *>(this->plugIn()[0]->getData())->getData();

    string str = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    graph->setYAxixLegend(str);

    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(graph);
}
COperator * OperatorSetYAxisPlot::clone(){
    return new OperatorSetYAxisPlot();
}
