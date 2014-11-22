#include "OperatorSetWidthPlot.h"

#include<CData.h>
#include<DataPlot.h>
#include<DataNumber.h>

#include "data/utility/PlotGraph.h"

OperatorSetWidthPlot::OperatorSetWidthPlot(){
        this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorSetWidthPlot");
    this->setName("width");
    this->setInformation("set the Width of the graph");
    this->structurePlug().addPlugIn(DataPlot::KEY,"g.plot");
    this->structurePlug().addPlugIn(DataNumber::KEY,"width.num");;
    this->structurePlug().addPlugOut(DataPlot::KEY,"f.plot");
}
void OperatorSetWidthPlot::exec(){
    shared_ptr<PlotGraph> graph = dynamic_cast<DataPlot *>(this->plugIn()[0]->getData())->getData();

    double width = dynamic_cast<DataNumber *>(this->plugIn()[1]->getData())->getValue();
    graph->VGraph()[0].setWidth(width);

    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(graph);
}
COperator * OperatorSetWidthPlot::clone(){
    return new OperatorSetWidthPlot();
}
