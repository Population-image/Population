#include "OperatorSetLegendPlot.h"

#include<CData.h>
#include<DataPlot.h>
#include<DataString.h>

#include "data/utility/PlotGraph.h"

OperatorSetLegendPlot::OperatorSetLegendPlot(){
        this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorSetLegendPlot");
    this->setName("legend");
    this->setInformation("set the Legend of the graph");
    this->structurePlug().addPlugIn(DataPlot::KEY,"g.plot");
    this->structurePlug().addPlugIn(DataString::KEY,"legend.str");;
    this->structurePlug().addPlugOut(DataPlot::KEY,"f.plot");
}
void OperatorSetLegendPlot::exec(){
    shared_ptr<PlotGraph> graph = dynamic_cast<DataPlot *>(this->plugIn()[0]->getData())->getData();

    string str = dynamic_cast<DataString *>(this->plugIn()[1]->getData())->getValue();
    graph->VGraph()[0].setLegend(str);

    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(graph);
}
COperator * OperatorSetLegendPlot::clone(){
    return new OperatorSetLegendPlot();
}
