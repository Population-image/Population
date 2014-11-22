#include "OperatorPopPointPlot.h"

#include<CData.h>
#include<DataPlot.h>
#include<DataNumber.h>

#include "data/utility/PlotGraph.h"

OperatorPopPointPlot::OperatorPopPointPlot(){
        this->path().push_back("Data");
    this->path().push_back("Plot");
    this->setKey("OperatorPopPointPlot");
    this->setName("popPoint");
    this->setInformation("Pop the first point of the graph");
    this->structurePlug().addPlugIn(DataPlot::KEY,"g.plot");
    this->structurePlug().addPlugOut(DataPlot::KEY,"f.plot");
}
void OperatorPopPointPlot::exec(){
    shared_ptr<PlotGraph> g = dynamic_cast<DataPlot *>(this->plugIn()[0]->getData())->getData();

    g->VGraph()[0].X().pop_front();
    g->VGraph()[0].Y().pop_front();
    dynamic_cast<DataPlot *>(this->plugOut()[0]->getData())->setData(g);
}
COperator * OperatorPopPointPlot::clone(){
    return new OperatorPopPointPlot();
}
